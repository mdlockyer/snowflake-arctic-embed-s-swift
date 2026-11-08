import Foundation
import Synchronization

/// File name of the quantized model pack inside a model directory (see `ArcticEmbedS.fromDir`).
public let MODEL_FILE: String = "model.ae8"
/// File name of the WordPiece vocabulary inside a model directory.
public let VOCAB_FILE: String = "vocab.txt"
/// Environment variable that overrides the model directory (`vocab.txt` + `model.ae8`).
public let MODEL_DIR_ENV_VAR: String = "ARCTIC_EMBED_MODEL_DIR"

/// Model directory used by the bundled binary and examples: `$ARCTIC_EMBED_MODEL_DIR` if set,
/// otherwise `./model` (the pack committed in this repository, when run from its root).
public func defaultModelDir() -> String {
    if let env = ProcessInfo.processInfo.environment[MODEL_DIR_ENV_VAR], !env.isEmpty {
        return env
    }
    return "model"
}

/// High-performance Pure-Swift Int8 inference engine for `Snowflake/snowflake-arctic-embed-s`.
public final class ArcticEmbedS: @unchecked Sendable {
    public let tokenizer: BertTokenizer
    public let model: PackedModel
    public let rawSizeBytes: Int

    public init(tokenizer: BertTokenizer, model: PackedModel, rawSizeBytes: Int) {
        self.tokenizer = tokenizer
        self.model = model
        self.rawSizeBytes = rawSizeBytes
    }

    /// Loads `vocab.txt` and the quantized model pack `model.ae8` from `dir`, and packs the weights
    /// for the fastest SIMD backend the host CPU supports (a few milliseconds).
    public static func fromDir(_ dir: String) throws -> ArcticEmbedS {
        try fromDirWithBackend(dir, backend: SimdBackend.detect())
    }

    public static func fromDir(_ dirURL: URL) throws -> ArcticEmbedS {
        try fromDirWithBackend(dirURL.path, backend: SimdBackend.detect())
    }

    /// Like `fromDir` with an explicit backend. The backend must be supported by the host CPU
    /// (`SimdBackend.isSupported`); otherwise this throws an error.
    public static func fromDirWithBackend(_ dir: String, backend: SimdBackend) throws -> ArcticEmbedS {
        guard backend.isSupported() else {
            throw ArcticEmbedError.unsupportedBackend(backend.name())
        }
        let dirURL = URL(fileURLWithPath: dir)
        let tokenizer = try BertTokenizer.fromVocabFile(dirURL.appendingPathComponent(VOCAB_FILE))
        let raw = try QuantizedModelRaw.loadAe8(dirURL.appendingPathComponent(MODEL_FILE))
        let model = PackedModel.fromRaw(raw, backend: backend)
        let rawSizeBytes = model.sizeBytes()
        return ArcticEmbedS(tokenizer: tokenizer, model: model, rawSizeBytes: rawSizeBytes)
    }

    /// Embeds a document/passage (`[CLS] + text + [SEP]`) using a reusable `Workspace`.
    @inlinable
    public func embedDocument(_ text: String, ws: Workspace) -> [Float] {
        var out = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        embedDocumentInto(text, ws: ws, out: &out)
        return out
    }

    @inlinable
    public func embedDocumentInto(_ text: String, ws: Workspace, out: inout [Float]) {
        tokenizer.encodeInto(text, isQuery: false, out: &ws.tokenIds)
        model.forwardCls(ws.tokenIds, ws: ws, outEmb: &out)
    }

    /// Embeds a search query (`[CLS] + QUERY_PREFIX + query + [SEP]`) using a reusable `Workspace`.
    @inlinable
    public func embedQuery(_ query: String, ws: Workspace) -> [Float] {
        var out = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        embedQueryInto(query, ws: ws, out: &out)
        return out
    }

    @inlinable
    public func embedQueryInto(_ query: String, ws: Workspace, out: inout [Float]) {
        tokenizer.encodeInto(query, isQuery: true, out: &ws.tokenIds)
        model.forwardCls(ws.tokenIds, ws: ws, outEmb: &out)
    }

    /// Embeds pre-tokenized `inputIds` using a reusable `Workspace`.
    @inlinable
    public func embedTokens(_ inputIds: [UInt16], ws: Workspace) -> [Float] {
        var out = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        model.forwardCls(inputIds, ws: ws, outEmb: &out)
        return out
    }

    @inlinable
    public func embedTokensInto(_ inputIds: UnsafeBufferPointer<UInt16>, ws: Workspace, out: UnsafeMutablePointer<Float>) {
        model.forwardCls(inputIds: inputIds, ws: ws, outEmb: out)
    }

    /// Low-latency single-query embedding using several cores for one sequence.
    public func embedQueryParallel(_ query: String, par: ParallelContext) -> [Float] {
        let ids = tokenizer.encodeQuery(query)
        return embedTokensParallel(ids, par: par)
    }

    /// Low-latency single-document embedding using several cores for one sequence.
    public func embedDocumentParallel(_ text: String, par: ParallelContext) -> [Float] {
        let ids = tokenizer.encode(text)
        return embedTokensParallel(ids, par: par)
    }

    /// Embeds pre-tokenized `inputIds` using several cores for one sequence.
    public func embedTokensParallel(_ inputIds: [UInt16], par: ParallelContext) -> [Float] {
        var out = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        par.forwardCls(model: model, inputIds: inputIds, outEmb: &out)
        return out
    }

    /// Multi-threaded lock-free work-stealing batch inference with unpadded ragged sequence packing ("Flash-VarLen").
    public func embedBatch(
        _ texts: [String],
        isQuery: Bool,
        numThreads: Int
    ) -> [[Float]] {
        let n = texts.count
        if n == 0 { return [] }
        let threads = min(max(numThreads, 1), n)
        let flatOut = AlignedBuf<Float>(zeroed: n * HIDDEN_SIZE)
        let nextIdx = PaddedCounter()

        let runWorker: @Sendable () -> Void = { [self] in
            let ws = Workspace()
            var tokBuf0: [UInt16] = []
            var tokBuf1: [UInt16] = []
            var tokBuf2: [UInt16] = []
            var tokBuf3: [UInt16] = []
            tokBuf0.reserveCapacity(64)
            tokBuf1.reserveCapacity(64)
            tokBuf2.reserveCapacity(64)
            tokBuf3.reserveCapacity(64)
            var batchIndices = (0, 0, 0, 0)
            let outStaging = AlignedBuf<Float>(zeroed: 4 * HIDDEN_SIZE)
            let MAX_PACK_TOKENS = 32
            let resultsPtr = flatOut.ptr
            let byteCount = HIDDEN_SIZE * MemoryLayout<Float>.stride

            while true {
                let idx0 = nextIdx.value.wrappingAdd(1, ordering: .relaxed).oldValue
                if idx0 >= n { break }

                self.tokenizer.encodeInto(texts[idx0], isQuery: isQuery, out: &tokBuf0)

                if tokBuf0.count >= 16 {
                    tokBuf0.withUnsafeBufferPointer { b0 in
                        self.model.forwardCls(
                            inputIds: b0,
                            ws: ws,
                            outEmb: resultsPtr.advanced(by: idx0 * HIDDEN_SIZE)
                        )
                    }
                    continue
                }

                batchIndices.0 = idx0
                var packCount = 1
                var packTokens = tokBuf0.count

                while packCount < 4 && packTokens + 12 <= MAX_PACK_TOKENS {
                    let nextI = nextIdx.value.wrappingAdd(1, ordering: .relaxed).oldValue
                    if nextI >= n { break }

                    switch packCount {
                    case 1:
                        self.tokenizer.encodeInto(texts[nextI], isQuery: isQuery, out: &tokBuf1)
                        if packTokens + tokBuf1.count <= MAX_PACK_TOKENS {
                            batchIndices.1 = nextI
                            packTokens += tokBuf1.count
                            packCount += 1
                        } else {
                            tokBuf1.withUnsafeBufferPointer { b1 in
                                self.model.forwardCls(
                                    inputIds: b1,
                                    ws: ws,
                                    outEmb: resultsPtr.advanced(by: nextI * HIDDEN_SIZE)
                                )
                            }
                            packTokens = MAX_PACK_TOKENS + 1 // break outer while
                        }
                    case 2:
                        self.tokenizer.encodeInto(texts[nextI], isQuery: isQuery, out: &tokBuf2)
                        if packTokens + tokBuf2.count <= MAX_PACK_TOKENS {
                            batchIndices.2 = nextI
                            packTokens += tokBuf2.count
                            packCount += 1
                        } else {
                            tokBuf2.withUnsafeBufferPointer { b2 in
                                self.model.forwardCls(
                                    inputIds: b2,
                                    ws: ws,
                                    outEmb: resultsPtr.advanced(by: nextI * HIDDEN_SIZE)
                                )
                            }
                            packTokens = MAX_PACK_TOKENS + 1
                        }
                    default:
                        self.tokenizer.encodeInto(texts[nextI], isQuery: isQuery, out: &tokBuf3)
                        if packTokens + tokBuf3.count <= MAX_PACK_TOKENS {
                            batchIndices.3 = nextI
                            packTokens += tokBuf3.count
                            packCount += 1
                        } else {
                            tokBuf3.withUnsafeBufferPointer { b3 in
                                self.model.forwardCls(
                                    inputIds: b3,
                                    ws: ws,
                                    outEmb: resultsPtr.advanced(by: nextI * HIDDEN_SIZE)
                                )
                            }
                            packTokens = MAX_PACK_TOKENS + 1
                        }
                    }
                }

                if packCount == 1 {
                    tokBuf0.withUnsafeBufferPointer { b0 in
                        self.model.forwardCls(
                            inputIds: b0,
                            ws: ws,
                            outEmb: resultsPtr.advanced(by: batchIndices.0 * HIDDEN_SIZE)
                        )
                    }
                } else {
                    tokBuf0.withUnsafeBufferPointer { b0 in
                        tokBuf1.withUnsafeBufferPointer { b1 in
                            tokBuf2.withUnsafeBufferPointer { b2 in
                                tokBuf3.withUnsafeBufferPointer { b3 in
                                    var seqRefs: [UnsafeBufferPointer<UInt16>] = []
                                    seqRefs.reserveCapacity(packCount)
                                    seqRefs.append(b0)
                                    if packCount >= 2 { seqRefs.append(b1) }
                                    if packCount >= 3 { seqRefs.append(b2) }
                                    if packCount >= 4 { seqRefs.append(b3) }
                                    self.model.forwardRaggedCls(
                                        sequences: seqRefs,
                                        ws: ws,
                                        outEmbs: outStaging.ptr
                                    )
                                }
                            }
                        }
                    }
                    let indicesArr = [batchIndices.0, batchIndices.1, batchIndices.2, batchIndices.3]
                    for k in 0..<packCount {
                        UnsafeMutableRawPointer(resultsPtr.advanced(by: indicesArr[k] * HIDDEN_SIZE)).copyMemory(
                            from: UnsafeRawPointer(outStaging.ptr.advanced(by: k * HIDDEN_SIZE)),
                            byteCount: byteCount
                        )
                    }
                }
            }
        }

        if threads == 1 {
            runWorker()
        } else {
            DispatchQueue.concurrentPerform(iterations: threads) { _ in
                runWorker()
            }
        }

        var results: [[Float]] = []
        results.reserveCapacity(n)
        for i in 0..<n {
            let start = flatOut.ptr.advanced(by: i * HIDDEN_SIZE)
            results.append(Array(UnsafeBufferPointer(start: start, count: HIDDEN_SIZE)))
        }
        return results
    }
}

/// Computes cosine similarity (dot product of L2-normalized 384-dim embeddings).
@inlinable
public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    var sum: Float = 0.0
    for i in 0..<HIDDEN_SIZE {
        sum += a[i] * b[i]
    }
    return sum
}
