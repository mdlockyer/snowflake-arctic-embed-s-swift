import ArcticEmbedS
import Foundation
import XCTest

final class ArcticEmbedSTests: XCTestCase {
    private func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func modelDir() -> URL {
        if let env = ProcessInfo.processInfo.environment[MODEL_DIR_ENV_VAR], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        return repoRoot().appendingPathComponent("model")
    }

    /// Returns `modelDir()` only when it holds both required files; otherwise throws `XCTSkip` so
    /// a fresh clone (where the gitignored `model/model.ae8` has not been fetched) reports a
    /// *skip* with the fix, instead of failing before any assertion. `MODEL_FILE` / `VOCAB_FILE`
    /// are the public file-name constants (ArcticEmbedS.swift:4-7).
    private func modelDirOrSkip() throws -> URL {
        let dir = modelDir()
        let fm = FileManager.default
        let modelURL = dir.appendingPathComponent(MODEL_FILE)
        let vocabURL = dir.appendingPathComponent(VOCAB_FILE)
        guard fm.fileExists(atPath: modelURL.path), fm.fileExists(atPath: vocabURL.path) else {
            throw XCTSkip("model pack missing in \(dir.path); run scripts/fetch_model.sh")
        }
        return dir
    }

    private func testDataDir() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("data")
    }

    struct RefRecord: Decodable {
        let kind: String
        let raw_text: String
        let ids: [UInt16]
        let embedding_fp32: [Float]
    }

    struct LongRecord: Decodable {
        let name: String
        let ids: [UInt16]
        let embedding_fp32: [Float]
    }

    /// Asserts the documented embedding contract on one produced embedding:
    /// `HIDDEN_SIZE` elements with unit L2 norm.
    ///
    /// oracle row 1: README.md:51 — "Embeddings are L2-normalized [Float] of length 384";
    /// Quantize.swift:3 — `HIDDEN_SIZE = 384`.
    /// oracle row 2: README.md:51; Model.swift:187-188 — `abs(norm - 1.0) <= 1e-4`, i.e.
    /// `sum(emb[i]*emb[i]) in [1-1e-4, 1+1e-4]`. The tolerance is the worst-case fp32
    /// serial-sum rounding over 384 terms (`384 * 2^-24 ~= 2.3e-5`).
    /// oracle row 3: ArcticEmbedS.swift:278-280 — `cosineSimilarity` is the raw dot product, so
    /// `cosineSimilarity(emb, emb)` is `||emb||^2`, which equals 1 iff the embedding is unit-norm.
    /// A broken `l2NormalizeInplace` leaves the norm at `sqrt(384) ~= 19.596` (self-cos 384) or
    /// `2.0` when double-scaled (self-cos 4); the two-sided window below rejects both.
    private func assertUnitEmbedding(
        _ emb: [Float],
        _ label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            emb.count,
            HIDDEN_SIZE, // oracle row 1: README.md:51/Quantize.swift:3 — 384 elements.
            "\(label): embedding must have HIDDEN_SIZE elements",
            file: file,
            line: line
        )
        let selfCos = cosineSimilarity(emb, emb)
        // oracle row 2/3: ||emb||^2 must be 1.0 within 1e-4 (A9).
        XCTAssertEqual(
            selfCos,
            1.0,
            accuracy: 1e-4, // oracle row 2 (A9): fp32 serial-sum bound 384*2^-24 ~= 2.3e-5.
            "\(label): embedding is not L2-normalized; cos(emb, emb) = \(selfCos), expected 1.0 +/- 1e-4",
            file: file,
            line: line
        )
    }

    func testTokenizerAndInt8AccuracyAgainstFp32Reference() throws {
        let refURL = testDataDir().appendingPathComponent("reference_test_vectors.json")
        let refData = try Data(contentsOf: refURL)
        let records = try JSONDecoder().decode([RefRecord].self, from: refData)
        XCTAssertEqual(records.count, 12)

        let engine = try ArcticEmbedS.fromDir(try modelDirOrSkip())
        let ws = Workspace()

        var cosSum: Float = 0.0
        var minCos: Float = 1.0

        for (idx, rec) in records.enumerated() {
            let isQuery = (rec.kind == "query")
            let gotIds = isQuery
                ? engine.tokenizer.encodeQuery(rec.raw_text)
                : engine.tokenizer.encode(rec.raw_text)
            XCTAssertEqual(
                gotIds,
                rec.ids,
                "Tokenizer mismatch on record \(idx) (\(rec.raw_text))"
            )

            let emb = isQuery
                ? engine.embedQuery(rec.raw_text, ws: ws)
                : engine.embedDocument(rec.raw_text, ws: ws)
            assertUnitEmbedding(emb, "record \(idx)")
            let cos = cosineSimilarity(emb, rec.embedding_fp32)
            let preview = String(rec.raw_text.prefix(45))
            print(String(
                format: "Record %2d (%@ | %@): cosine_sim vs FP32 = %.6f",
                idx, rec.kind, preview, cos
            ))
            // 0.9985: pre-existing per-record floor (oracle U10 — README:7-8 gives only round
            // figures, so the exact floor is the test's choice; kept unchanged).
            XCTAssertGreaterThan(cos, 0.9985, "Cosine similarity too low on record \(idx): \(cos)")
            // oracle row 4: for unit-norm a and b, a.b <= 1 (attained iff a == b, up to fp32
            // accumulation of 384*2^-24 ~= 2.3e-5); 1.0001 is the safe ceiling. A one-sided lower
            // bound alone is satisfied by any positive rescaling of the embedding (oracle row 5).
            XCTAssertLessThanOrEqual(cos, 1.0001, "Cosine similarity above the unit-vector ceiling on record \(idx): \(cos)")
            cosSum += cos
            if cos < minCos {
                minCos = cos
            }
        }

        let meanCos = cosSum / Float(records.count)
        print(String(
            format: "Summary across %d reference vectors: mean_cos = %.6f, min_cos = %.6f",
            records.count, meanCos, minCos
        ))
        // 0.9992: pre-existing mean floor (oracle U10 — the test's choice; kept unchanged).
        XCTAssertGreaterThan(meanCos, 0.9992, "Mean cosine similarity below 0.9992: \(meanCos)")
        // oracle row 4/5: the mean of unit-vector cosines cannot exceed 1.0001.
        XCTAssertLessThanOrEqual(meanCos, 1.0001, "Mean cosine similarity above the unit-vector ceiling: \(meanCos)")

        // Verify that unpadded ragged sequence packing (`embedBatch` / `forwardRaggedCls`) matches single-sequence inference
        let docTexts = records.filter { $0.kind == "document" }.map { $0.raw_text }
        let batchEmbs = engine.embedBatch(docTexts, isQuery: false, numThreads: 1)
        for (i, text) in docTexts.enumerated() {
            let singleEmb = engine.embedDocument(text, ws: ws)
            assertUnitEmbedding(batchEmbs[i], "batch doc \(i)")
            assertUnitEmbedding(singleEmb, "single doc \(i)")
            let cosRagged = cosineSimilarity(singleEmb, batchEmbs[i])
            // oracle row 11: `embedBatch[i]` is the embedding of `texts[i]`; 0.99999 lower bound.
            XCTAssertGreaterThan(
                cosRagged,
                0.99999,
                "Ragged batch mismatch at doc \(i): cos=\(cosRagged)"
            )
            // oracle row 11/4: both sides are unit-norm, so the cosine also cannot exceed 1.0001.
            XCTAssertLessThanOrEqual(
                cosRagged,
                1.0001,
                "Ragged batch cosine above the unit-vector ceiling at doc \(i): cos=\(cosRagged)"
            )
        }
    }

    /// Filler vocabulary for the packing-regimes test. Every word is a single WordPiece root
    /// token in the tracked `model/vocab.txt`, so `[CLS] + words + [SEP]` has `wordCount + 2`
    /// tokens (A10) without any packer constant being pinned (U1/U2).
    private let fillerWords = [
        "data", "cloud", "query", "search", "vector", "index",
        "model", "passage", "text", "table", "stream", "fast",
    ]

    private func fillerText(_ wordCount: Int) -> String {
        (0..<wordCount).map { fillerWords[$0 % fillerWords.count] }.joined(separator: " ")
    }

    /// `embedBatch` must return, for every input, the same embedding as the single-sequence path,
    /// across packing regimes: several very short documents (multi-sequence packs), mid-length
    /// documents, four 100+-word documents (forwarded standalone), and a duplicate text. The
    /// batch is deliberately diverse; the packing limits are not public (U1/U2, oracle rows 15/16),
    /// so the assertions are the derivable parity contract, never an internal `packCount`/token
    /// total.
    func testRaggedBatchMatchesSingleSequenceAcrossPackingRegimes() throws {
        let engine = try ArcticEmbedS.fromDir(try modelDirOrSkip())
        let ws = Workspace()

        // 8 very short docs (2-4 words => 4-6 tokens), a 12-word mid doc first so the short run
        // that follows fills one pack, then more shorts, three mid docs (10-14 words), four long
        // docs (100+ words each => always standalone, 102+ tokens) and one duplicate of `texts[0]`.
        var texts: [String] = []
        texts.append(fillerText(4)) // idx 0: 6 tokens
        texts.append(fillerText(4)) // idx 1: 6 tokens
        texts.append(fillerText(4)) // idx 2: 6 tokens
        texts.append(fillerText(12)) // idx 3: 14 tokens (mid)
        texts.append(fillerText(3)) // idx 4: 5 tokens (short)
        texts.append(fillerText(3)) // idx 5: 5 tokens (short)
        texts.append(fillerText(2)) // idx 6: 4 tokens (short)
        texts.append(fillerText(3)) // idx 7: 5 tokens (short)
        texts.append(fillerText(14)) // idx 8: 16 tokens (mid, standalone)
        texts.append(fillerText(10)) // idx 9: 12 tokens (mid)
        texts.append(fillerText(13)) // idx 10: 15 tokens (mid)
        texts.append(fillerText(100)) // idx 11: long
        texts.append(fillerText(102)) // idx 12: long
        texts.append(fillerText(101)) // idx 13: long
        texts.append(fillerText(103)) // idx 14: long
        texts.append(texts[0]) // idx 15: duplicate of idx 0

        // Single-threaded document batch.
        let docBatch = engine.embedBatch(texts, isQuery: false, numThreads: 1)
        // oracle row 14: one embedding per input text.
        XCTAssertEqual(docBatch.count, texts.count)

        for (i, text) in texts.enumerated() {
            let single = engine.embedDocument(text, ws: ws)
            assertUnitEmbedding(docBatch[i], "batch doc \(i)")
            assertUnitEmbedding(single, "single doc \(i)")
            let cos = cosineSimilarity(docBatch[i], single)
            // oracle row 11: `embedBatch[i]` is documented as the embedding of `texts[i]`;
            // packing may not change the value. 0.99999 lower bound.
            XCTAssertGreaterThan(cos, 0.99999, "ragged batch mismatch at doc \(i): cos=\(cos)")
            // oracle row 11/4: both sides are unit-norm, so the cosine cannot exceed 1.0001.
            XCTAssertLessThanOrEqual(cos, 1.0001, "ragged batch cosine above the unit-vector ceiling at doc \(i): cos=\(cos)")
            // oracle row 11 derivation: cos >= 0.99999 => ||delta|| <= sqrt(2*(1-0.99999)) =
            // sqrt(2e-5) = 4.47e-3, and no element can exceed the norm => maxDiff <= 4.47e-3.
            let maxDiff = zip(docBatch[i], single).map { abs($0 - $1) }.max() ?? 0.0
            print(String(format: "ragged batch vs single doc %2d: cos=%.8f maxDiff=%.2e", i, cos, maxDiff))
            XCTAssertLessThanOrEqual(maxDiff, 4.47e-3, "ragged batch element divergence at doc \(i): maxDiff=\(maxDiff)")
            // The approved production contract is stronger than an accuracy floor: packing and
            // neighbour content must not change any component of a document's embedding.
            XCTAssertEqual(docBatch[i], single, "pack neighbours changed the embedding at doc \(i)")
        }

        // Query batch on four worker threads.
        let queryBatch = engine.embedBatch(texts, isQuery: true, numThreads: 4)
        // oracle row 14: one embedding per input text, regardless of `numThreads`.
        XCTAssertEqual(queryBatch.count, texts.count)
        for (i, text) in texts.enumerated() {
            let singleQuery = engine.embedQuery(text, ws: ws)
            assertUnitEmbedding(queryBatch[i], "query batch doc \(i)")
            let cos = cosineSimilarity(queryBatch[i], singleQuery)
            // oracle row 12: `isQuery: true` selects exactly the `embedQuery` tokenisation, so the
            // outputs are the same embedding; bounds as in oracle row 11.
            XCTAssertGreaterThan(cos, 0.99999, "query batch mismatch at doc \(i): cos=\(cos)")
            XCTAssertLessThanOrEqual(cos, 1.0001, "query batch cosine above the unit-vector ceiling at doc \(i): cos=\(cos)")
            let maxDiff = zip(queryBatch[i], singleQuery).map { abs($0 - $1) }.max() ?? 0.0
            XCTAssertLessThanOrEqual(maxDiff, 4.47e-3, "query batch element divergence at doc \(i): maxDiff=\(maxDiff)")
            XCTAssertEqual(queryBatch[i], singleQuery, "pack neighbours changed the query embedding at doc \(i)")
            // oracle row 12: the query path prepends QUERY_PREFIX, so a non-empty text cannot
            // yield the same vector as a document.
            XCTAssertNotEqual(queryBatch[i], docBatch[i], "query and document embeddings are identical at doc \(i)")
        }

        // Thread-count invariance: `numThreads` is a parallelism knob, not a model input
        // (oracle row 13, A5), so the 4-thread document batch is element-wise identical.
        let docBatchMT = engine.embedBatch(texts, isQuery: false, numThreads: 4)
        XCTAssertEqual(docBatchMT.count, texts.count)
        for i in texts.indices {
            assertUnitEmbedding(docBatchMT[i], "4-thread batch doc \(i)")
            print(String(format: "docBatch(4 threads) vs docBatch(1 thread) doc %2d: cos=%.6f", i, cosineSimilarity(docBatchMT[i], docBatch[i])))
            XCTAssertEqual(docBatchMT[i], docBatch[i], "numThreads changed the result at doc \(i)")
        }

        // oracle row 14: one embedding per input text; empty in, empty out.
        XCTAssertTrue(engine.embedBatch([], isQuery: false, numThreads: 8).isEmpty, "empty batch must return no embeddings")
    }

    /// A direct ragged call with sequences long enough to use SME attention individually must
    /// produce exactly the same embeddings as three single-sequence forwards. The output layout
    /// is established by the existing caller at ArcticEmbedS.swift:249-254, which copies slice k
    /// to input sequence k (resolving oracle U9 from a caller rather than an internal layout).
    func testForwardRaggedClsMatchesSingleSequenceForLongSegments() throws {
        let engine = try ArcticEmbedS.fromDir(try modelDirOrSkip())
        let ws = Workspace()
        let lengths = [16, 20, 33]
        let ids = lengths.map { length in
            // oracle A10: these single-token filler words yield wordCount + 2 ids.
            let encoded = engine.tokenizer.encode(fillerText(length - 2))
            XCTAssertEqual(encoded.count, length)
            return encoded
        }
        var packed = [Float](repeating: 0.0, count: ids.count * HIDDEN_SIZE)

        ids[0].withUnsafeBufferPointer { b0 in
            ids[1].withUnsafeBufferPointer { b1 in
                ids[2].withUnsafeBufferPointer { b2 in
                    packed.withUnsafeMutableBufferPointer { out in
                        engine.model.forwardRaggedCls(
                            sequences: [b0, b1, b2],
                            ws: ws,
                            outEmbs: out.baseAddress!
                        )
                    }
                }
            }
        }

        for i in ids.indices {
            let single = engine.embedTokens(ids[i], ws: ws)
            let ragged = Array(packed[(i * HIDDEN_SIZE)..<((i + 1) * HIDDEN_SIZE)])
            assertUnitEmbedding(ragged, "direct ragged segment \(i)")
            // Batch parity is the behavioral contract; unlike a tolerance, exact equality also
            // proves that no pack-wide quantization scale can depend on a neighbouring segment.
            XCTAssertEqual(ragged, single, "direct ragged segment \(i) diverged from forwardCls")
        }
    }

    func testAllSimdBackendsMatchScalar() throws {
        let dir = try modelDirOrSkip()
        let raw = try QuantizedModelRaw.loadAe8(dir.appendingPathComponent(MODEL_FILE))
        let tok = try BertTokenizer.fromVocabFile(dir.appendingPathComponent(VOCAB_FILE))

        let ids = tok.encodeQuery("what is snowflake?")
        let ws = Workspace()

        var scalarOut = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        PackedModel.fromRaw(raw, backend: .scalar).forwardCls(ids, ws: ws, outEmb: &scalarOut)
        assertUnitEmbedding(scalarOut, "scalar")

        var testedBackendCount = 0
        for b in [
            SimdBackend.armSme,
            SimdBackend.armI8mm,
            SimdBackend.armDotProd,
            SimdBackend.armNeon,
            SimdBackend.x86AvxVnni,
            SimdBackend.x86Avx2,
        ] {
            if !b.isSupported() {
                print("Backend \(b.name()) not supported on this CPU, skipped")
                continue
            }
            testedBackendCount += 1
            var out = [Float](repeating: 0.0, count: HIDDEN_SIZE)
            PackedModel.fromRaw(raw, backend: b).forwardCls(ids, ws: ws, outEmb: &out)
            assertUnitEmbedding(out, "backend \(b.name())")
            let cos = cosineSimilarity(scalarOut, out)
            let maxDiff = zip(scalarOut, out).map { abs($0 - $1) }.max() ?? 0.0
            print(String(
                format: "Backend %@ vs Scalar: cos=%.8f, max_diff=%.8f",
                b.name(), cos, maxDiff
            ))
            // 0.999: pre-existing backend-parity floor (oracle C-2 — this is not the README FP32
            // accuracy claim; kept unchanged).
            XCTAssertGreaterThan(cos, 0.999, "Backend \(b) diverged from Scalar: cos=\(cos)")
            // oracle row 4: both operands are unit-norm, so the raw dot product cannot exceed
            // 1.0001 (fp32 accumulation ceiling); a one-sided bound accepts rescaled output.
            XCTAssertLessThanOrEqual(cos, 1.0001, "Backend \(b) cosine above the unit-vector ceiling: cos=\(cos)")
            // oracle row 9 (A2): the necessary window for anything meeting the README cosine floor
            // is [2.28e-3, 4.47e-2]; 0.01 sits inside it with 1.7x margin over the observed
            // max_diff = 5.81e-3. `maxDiff` was computed but never asserted before.
            XCTAssertLessThan(maxDiff, 0.01, "Backend \(b) diverged from Scalar: max_diff=\(maxDiff)")
        }

        // oracle row 17: `.scalar` is the portable fallback and is always supported.
        XCTAssertTrue(SimdBackend.scalar.isSupported(), "the scalar fallback must always be supported")
        // oracle row 18: `fromDir` packs for the fastest backend the host supports, so `detect()`
        // must be a supported backend.
        XCTAssertTrue(SimdBackend.detect().isSupported(), "detect() returned an unsupported backend")
        #if arch(arm64)
        // oracle row 17: NEON is architectural baseline on ARMv8-A/arm64, so at least one SIMD
        // backend must run here. Without this guard an all-false isSupported() makes the loop
        // vacuous (the recorded mutant).
        XCTAssertGreaterThan(testedBackendCount, 0, "no SIMD backend was exercised; the parity loop was vacuous")
        XCTAssertTrue(SimdBackend.armNeon.isSupported(), "NEON is architectural baseline on arm64")
        #endif
        // x86_64 has no mandatory SIMD baseline: runtime selection may legitimately choose the
        // documented scalar fallback on a CPU without AVX2 or AVX-VNNI.

        // oracle row 29: `QuantizedLayerRaw.qOnly`/`kvOnly` doc comment (Quantize.swift:75-77) —
        // "Separate Q (`[384, 384]`) and KV (`[768, 384]`) for Layer 11 CLS-only pruning."
        // This is the documented, publicly observable shape of the pruning.
        XCTAssertNotNil(raw.layers[11].qOnly, "layer 11 must carry the split Q matrix")
        XCTAssertNotNil(raw.layers[11].kvOnly, "layer 11 must carry the split KV matrix")
        XCTAssertEqual(raw.layers[11].qOnly?.nOut, HIDDEN_SIZE, "layer 11 Q nOut = 384")
        XCTAssertEqual(raw.layers[11].qOnly?.kIn, HIDDEN_SIZE, "layer 11 Q kIn = 384")
        XCTAssertEqual(raw.layers[11].kvOnly?.nOut, 2 * HIDDEN_SIZE, "layer 11 KV nOut = 2*384 = 768")
        XCTAssertEqual(raw.layers[11].kvOnly?.kIn, HIDDEN_SIZE, "layer 11 KV kIn = 384")
        // oracle row 29: only layer 11 is split; layers 0...10 carry no qOnly/kvOnly.
        for layerIndex in 0..<11 {
            XCTAssertNil(raw.layers[layerIndex].qOnly, "layer \(layerIndex) must not carry a split Q matrix")
            XCTAssertNil(raw.layers[layerIndex].kvOnly, "layer \(layerIndex) must not carry a split KV matrix")
        }

        // oracle row 30: the pruned pack must still be a working engine — L2-normalized 384-dim
        // `[CLS]` embedding that matches Scalar on the documented backend-parity window.
        // The old `packedDetected.layers[11].qkv.packedW.count == 0` check is UNDERIVABLE
        // (oracle row 33 / U6: no doc comment specifies the dummy `qkv` when qOnly/kvOnly are
        // populated, and an equivalent pack may alias `qkv` to `qOnly`), and `sizeBytes()` is not a
        // pruning discriminator (oracle row 32: qOnly + kvOnly = qkv, so pruning leaves the size
        // unchanged). The derivable contracts are the loader shape above and the forward output
        // below.
        let packedDetected = PackedModel.fromRaw(raw, backend: SimdBackend.detect())
        var detectedOut = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        packedDetected.forwardCls(ids, ws: ws, outEmb: &detectedOut)
        assertUnitEmbedding(detectedOut, "detected backend (\(packedDetected.backend.name()))")
        let detectedCos = cosineSimilarity(scalarOut, detectedOut)
        let detectedMaxDiff = zip(scalarOut, detectedOut).map { abs($0 - $1) }.max() ?? 0.0
        print(String(
            format: "Detected backend %@ vs Scalar: cos=%.8f, max_diff=%.8f",
            packedDetected.backend.name(), detectedCos, detectedMaxDiff
        ))
        // oracle row 30: `cos >= 0.999` against Scalar (same backend-parity floor as the loop).
        XCTAssertGreaterThan(detectedCos, 0.999, "detected backend diverged from Scalar: cos=\(detectedCos)")
        // oracle row 30/4: both outputs are unit-norm, so the cosine cannot exceed 1.0001.
        XCTAssertLessThanOrEqual(detectedCos, 1.0001, "detected backend cosine above the unit-vector ceiling: cos=\(detectedCos)")
        // oracle row 30/9 (A2): same `maxDiff < 0.01` window as the backend loop.
        XCTAssertLessThan(detectedMaxDiff, 0.01, "detected backend diverged from Scalar: max_diff=\(detectedMaxDiff)")
    }

    /// Temporary directory for loader-rejection fixtures.
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("arctic-embed-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// The `.ae8` loader must reject a corrupted header and a truncated payload with the documented
    /// error case and description, and `fromDirWithBackend` must reject an unsupported backend
    /// before touching the filesystem.
    func testLoaderRejectsCorruptAndTruncatedPacks() throws {
        let tempDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        // MODEL_FILE is the public pack file-name constant (ArcticEmbedS.swift:4-5).
        let modelDirectory = try modelDirOrSkip()
        let source = modelDirectory.appendingPathComponent(MODEL_FILE)
        let original = try Data(contentsOf: source)

        // oracle row 20: corrupt the leading magic byte instead of hard-coding the private 8-byte
        // literal; the loader must report `.invalidMagic` with its fixed description.
        var corrupt = original
        corrupt[0] ^= 0xFF
        let corruptURL = tempDir.appendingPathComponent("corrupt.ae8")
        try corrupt.write(to: corruptURL)
        XCTAssertThrowsError(try QuantizedModelRaw.loadAe8(corruptURL), "a corrupted magic header must be rejected") { error in
            guard let arcticError = error as? ArcticEmbedError, case .invalidMagic = arcticError else {
                return XCTFail("expected ArcticEmbedError.invalidMagic, got \(error)")
            }
            // oracle row 20: description "invalid .ae8 magic header" (Quantize.swift:24-25).
            XCTAssertEqual(arcticError.description, "invalid .ae8 magic header")
        }

        // oracle row 21: a valid header followed by a truncated payload (first 1 MiB of the pack;
        // 1 MiB = 1024 * 1024 = 1_048_576 bytes) must report `.unexpectedEOF`, not `.invalidMagic`.
        let truncatedURL = tempDir.appendingPathComponent("truncated.ae8")
        try original.prefix(1_048_576).write(to: truncatedURL)
        XCTAssertThrowsError(try QuantizedModelRaw.loadAe8(truncatedURL), "a truncated payload must be rejected") { error in
            guard let arcticError = error as? ArcticEmbedError, case .unexpectedEOF = arcticError else {
                return XCTFail("expected ArcticEmbedError.unexpectedEOF, got \(error)")
            }
            // oracle row 21: description "unexpected end of file while reading model data".
            XCTAssertEqual(arcticError.description, "unexpected end of file while reading model data")
        }

        #if arch(arm64)
        // oracle row 17: x86_64 kernels are not supported on arm64. oracle row 19: the loader must
        // throw `.unsupportedBackend` (fromDirWithBackend checks support before reading any file).
        XCTAssertFalse(SimdBackend.x86Avx2.isSupported(), "x86_64 AVX2 cannot be supported on arm64")
        XCTAssertThrowsError(
            try ArcticEmbedS.fromDirWithBackend(modelDirectory.path, backend: .x86Avx2),
            "an unsupported backend must be rejected"
        ) { error in
            guard let arcticError = error as? ArcticEmbedError, case .unsupportedBackend = arcticError else {
                return XCTFail("expected ArcticEmbedError.unsupportedBackend, got \(error)")
            }
            // oracle row 19 / U8: assert the documented fixed template, not the undocumented name().
            XCTAssertTrue(arcticError.description.hasPrefix("SIMD backend "), "description: \(arcticError.description)")
            XCTAssertTrue(arcticError.description.hasSuffix(" is not supported by this CPU"), "description: \(arcticError.description)")
        }
        #endif
    }

    /// The loader and the engine report the documented pack footprint (README.md:13 "33.8 MB").
    func testLoaderReportsDocumentedPackSize() throws {
        let dir = try modelDirOrSkip()
        let raw = try QuantizedModelRaw.loadAe8(dir.appendingPathComponent(MODEL_FILE))
        // oracle row 24 derivation: word emb 30522*384 = 11,720,448 Int8 bytes + pos emb
        // 512*384 = 196,608 + 12 layers * 1,769,472 = 21,233,664 => 33,150,720 Int8 bytes, plus
        // ~0.53 MB of float scale/bias/LN sidecars (~33.68 MB) and the layer-11 split copy. The
        // 33_000_000 floor is valid for both the quantized footprint and the packed footprint;
        // 40_000_000 is a sanity ceiling for duplicated buffers (A3).
        XCTAssertGreaterThan(raw.sizeBytes(), 33_000_000, "quantized pack footprint is below the analytic floor")
        XCTAssertLessThan(raw.sizeBytes(), 40_000_000, "quantized pack footprint suggests duplicated buffers (A3)")

        let engine = try ArcticEmbedS.fromDir(dir)
        // oracle row 24/C-3: `rawSizeBytes` is a footprint, so only the shared lower bound is safe.
        XCTAssertGreaterThan(engine.rawSizeBytes, 33_000_000, "packed footprint is below the analytic floor")
    }

    /// `long_texts.txt` is committed alongside the pre-tokenized `long_reference_vectors.json`;
    /// encoding each raw line must reproduce that record's ids exactly. Only line 7 exceeds
    /// `MAX_SEQ_LEN`, so this is the tokenizer's truncation oracle.
    ///
    /// Tokenizer-only check against the **tracked** `model/vocab.txt`: no `model.ae8` is needed,
    /// so the truncation path stays covered in a fresh clone.
    ///
    /// oracle rows 1-7: `long_texts.txt` line *i* is the raw text of record `long{i}`
    /// (`long0`...`long6`), in file order; oracle row 8: 7 non-empty lines, one per record.
    func testLongTextFixtureEncodesToReferenceIds() throws {
        // oracle row 8: `long_texts.txt` holds 7 newline-terminated paragraphs (trailing newline
        // after line 7), so splitting on "\n" and discarding the empty tail yields 7 texts.
        let textsURL = testDataDir().appendingPathComponent("long_texts.txt")
        let textsData = try Data(contentsOf: textsURL)
        let lines = String(decoding: textsData, as: UTF8.self)
            .split(separator: "\n")
            .map(String.init)
        XCTAssertEqual(lines.count, 7, "long_texts.txt must hold 7 non-empty lines (oracle row 8)")

        let refURL = testDataDir().appendingPathComponent("long_reference_vectors.json")
        let records = try JSONDecoder().decode([LongRecord].self, from: try Data(contentsOf: refURL))
        XCTAssertEqual(records.count, 7, "long_reference_vectors.json must hold 7 records (oracle row 8)")

        // The tracked vocab ships with the repository (unlike the gitignored model.ae8).
        let vocabURL = repoRoot().appendingPathComponent("model").appendingPathComponent(VOCAB_FILE)
        let tokenizer = try BertTokenizer.fromVocabFile(vocabURL)

        for (line, rec) in zip(lines, records) {
            // oracle rows 1-7: encode(line_i) == long{i}.ids exactly (lengths 124, 228, 296, 431,
            // 450, 476, 512). Line 7 tokenizes to 924 ids before truncation, so this assertion
            // fails if the MAX_SEQ_LEN branch is removed (it would yield 924 ids, not 512).
            XCTAssertEqual(
                tokenizer.encode(line),
                rec.ids,
                "encode(long_texts.txt line for \(rec.name)) must match the reference ids"
            )
        }

        // oracle row 15: long6 is the only truncating input. Its 924-id stream
        // (448 line-5 + 474 line-6 text tokens + CLS + SEP) becomes [CLS] + first 510 text tokens
        // + [SEP] = exactly 512 ids.
        let long6 = records[6]
        // oracle row 8: file order is record order, so the last line pairs with `long6`.
        XCTAssertEqual(long6.name, "long6", "the 7th record must be long6")
        XCTAssertEqual(long6.ids.count, 512, "long6 must be truncated to MAX_SEQ_LEN = 512 ids (oracle row 15)")
        // oracle row 15: ids[0] = [CLS] = 101.
        XCTAssertEqual(long6.ids[0], CLS_ID, "long6 must open with [CLS]")
        // oracle row 15/16: [SEP] is appended *after* truncation, so it survives as the final id
        // (`[CLS] t1 ... t510 [SEP]`, never `[CLS] ... t512` with the SEP lost).
        XCTAssertEqual(long6.ids[511], SEP_ID, "long6 must close with [SEP] appended after truncation")
        // oracle row 15: the kept 510 text tokens are line-5's 448 tokens followed by line-6's
        // first 62; line-6 tokens #61/#62 are long5.ids[61..62] = 2408 ("across"), 1996 ("the").
        XCTAssertEqual(long6.ids[509], 2408, "long6.ids[509] = line-6 token #61 \"across\"")
        XCTAssertEqual(long6.ids[510], 1996, "long6.ids[510] = line-6 token #62 \"the\"")
    }

    func testLongSequenceAccuracyAgainstFp64Reference() throws {
        let dataURL = testDataDir().appendingPathComponent("long_reference_vectors.json")
        let data = try Data(contentsOf: dataURL)
        let records = try JSONDecoder().decode([LongRecord].self, from: data)
        XCTAssertEqual(records.count, 7)

        let engine = try ArcticEmbedS.fromDir(try modelDirOrSkip())
        let ws = Workspace()
        var sum: Float = 0.0
        // oracle row 4 rationale: start at +infinity so `min(minCos, cos)` is a true minimum.
        // Initialising to 1.0 made the summary vacuously `>= 1` whenever every cosine was
        // rescaled above 1 (the broken-`l2NormalizeInplace` case).
        var minCos: Float = .infinity

        for rec in records {
            let emb = engine.embedTokens(rec.ids, ws: ws)
            assertUnitEmbedding(emb, "long record \(rec.name)")
            let cos = cosineSimilarity(emb, rec.embedding_fp32)
            print(String(format: "%@ (%3d tokens): cosine_sim vs FP64 = %.6f", rec.name, rec.ids.count, cos))
            // oracle row 4: unit-norm operands bound the raw dot product above by 1.0001.
            XCTAssertLessThanOrEqual(cos, 1.0001, "Long record \(rec.name) cosine above the unit-vector ceiling: \(cos)")
            sum += cos
            minCos = min(minCos, cos)
        }
        let mean = sum / Float(records.count)
        print(String(format: "Long-sequence summary: mean_cos = %.6f, min_cos = %.6f", mean, minCos))
        // 0.998 / 0.999: pre-existing floors (oracle U10 — the test's choice; kept unchanged).
        XCTAssertGreaterThan(minCos, 0.998, "long-sequence min cosine too low: \(minCos)")
        XCTAssertGreaterThan(mean, 0.999, "long-sequence mean cosine too low: \(mean)")
        // oracle row 4/5: neither the min nor the mean of unit-vector cosines can exceed 1.0001.
        XCTAssertLessThanOrEqual(minCos, 1.0001, "long-sequence min cosine above the unit-vector ceiling: \(minCos)")
        XCTAssertLessThanOrEqual(mean, 1.0001, "long-sequence mean cosine above the unit-vector ceiling: \(mean)")
    }

    func testParallelForwardMatchesSingleThread() throws {
        let engine = try ArcticEmbedS.fromDir(try modelDirOrSkip())
        if engine.model.backend != .armSme {
            // A silent `return` would make this test assert nothing on non-SME CPUs; report the
            // missing coverage as a skip instead (oracle row 10 is SME-only).
            throw XCTSkip("intra-op parallel path needs the SME backend (have \(engine.model.backend.name()))")
        }
        let ws = Workspace()
        let lengths = [1, 2, 5, 8, 15, 16, 17, 31, 33, 64, 100, 128, 257, 512]
        for threads in [2, 3, 4] {
            let par = ParallelContext(nThreads: threads)
            for len in lengths {
                var ids: [UInt16] = [101]
                for i in 0..<max(len - 2, 0) {
                    ids.append(UInt16(1996 + ((i * 37) % 5000)))
                }
                if len >= 2 {
                    ids.append(102)
                }
                let single = engine.embedTokens(ids, ws: ws)
                _ = engine.embedTokensParallel(ids, par: par)
                let multi = engine.embedTokensParallel(ids, par: par)
                assertUnitEmbedding(single, "single threads=\(threads) len=\(len)")
                assertUnitEmbedding(multi, "parallel threads=\(threads) len=\(len)")
                let cos = cosineSimilarity(single, multi)
                let maxDiff = zip(single, multi).map { abs($0 - $1) }.max() ?? 0.0
                print(String(format: "threads=%d len=%3d: cos=%.9f max_diff=%.2e", threads, len, cos, maxDiff))
                // oracle row 10: bit-identical output is the documented contract; `cos` of two
                // unit vectors is then 1.0, and a one-sided bound alone would accept a rescaled copy.
                XCTAssertGreaterThan(
                    cos,
                    0.99999,
                    "parallel diverged: threads=\(threads) len=\(len) cos=\(cos)"
                )
                XCTAssertLessThanOrEqual(
                    cos,
                    1.0001,
                    "parallel cosine above the unit-vector ceiling: threads=\(threads) len=\(len) cos=\(cos)"
                )
                // oracle row 10: README.md:78 — "`ParallelContext` output is bit-identical to the
                // single-threaded forward pass (SME only)" => every element difference is exactly 0.
                // `maxDiff` was printed but never asserted before.
                XCTAssertLessThanOrEqual(
                    maxDiff,
                    0.0,
                    "parallel output is not bit-identical: threads=\(threads) len=\(len) max_diff=\(maxDiff)"
                )
            }
        }
    }

    /// The public low-latency wrappers must return exactly the single-threaded embeddings, and
    /// the query and document tokenization paths must stay distinct.
    func testParallelPublicAPIWrappersMatchSingleThread() throws {
        let engine = try ArcticEmbedS.fromDir(try modelDirOrSkip())
        let ws = Workspace()
        let text = "what is snowflake?"
        for nThreads in [1, 2, 3, 4] {
            let par = ParallelContext(nThreads: nThreads)
            // oracle row 25: `embedQueryParallel` sends the same query tokenisation through the
            // forward pass, and Parallel.swift:114-116 falls back to the single-threaded forward
            // pass off-SME; README:78 makes the multi-core path bit-identical => equal arrays.
            let querySingle = engine.embedQuery(text, ws: ws)
            let queryParallel = engine.embedQueryParallel(text, par: par)
            assertUnitEmbedding(queryParallel, "embedQueryParallel nThreads=\(nThreads)")
            XCTAssertEqual(
                queryParallel,
                querySingle,
                "embedQueryParallel(nThreads: \(nThreads)) diverged: cos=\(cosineSimilarity(queryParallel, querySingle))"
            )
            // oracle row 26: same reasoning with the document tokenisation (no prefix).
            let docSingle = engine.embedDocument(text, ws: ws)
            let docParallel = engine.embedDocumentParallel(text, par: par)
            assertUnitEmbedding(docParallel, "embedDocumentParallel nThreads=\(nThreads)")
            XCTAssertEqual(
                docParallel,
                docSingle,
                "embedDocumentParallel(nThreads: \(nThreads)) diverged: cos=\(cosineSimilarity(docParallel, docSingle))"
            )
            // oracle row 27: `embedQuery` prepends QUERY_PREFIX, so the two token sequences differ
            // and the resulting vectors cannot be equal for a non-empty text.
            XCTAssertNotEqual(
                queryParallel,
                docParallel,
                "query and document parallel embeddings are identical (nThreads: \(nThreads))"
            )
        }

    }

    /// `ParallelContext` recommends the documented number of worker threads for the active backend.
    func testParallelContextRecommendedThreadPolicy() throws {
        let engine = try ArcticEmbedS.fromDir(try modelDirOrSkip())
        // oracle row 28: "2 threads (one per P-cluster SME unit) when the SME backend is active,
        // otherwise 1 (plain single-threaded forward pass, no helper threads)".
        let expectedThreads = engine.model.backend == .armSme ? 2 : 1
        XCTAssertEqual(
            ParallelContext.recommended(engine).numThreads(),
            expectedThreads,
            "recommended() must pick 2 threads for SME and 1 otherwise"
        )
        // oracle row 28: `numThreads()` returns the value passed to `init(nThreads:)`.
        XCTAssertEqual(ParallelContext(nThreads: 3).numThreads(), 3)
    }

    /// Builds a tokenizer from the tracked vocabulary, with no gitignored model pack required.
    private func makeOfflineTokenizer() throws -> BertTokenizer {
        let vocabURL = repoRoot().appendingPathComponent("model").appendingPathComponent(VOCAB_FILE)
        return try BertTokenizer.fromVocabFile(vocabURL)
    }

    /// The 100-character WordPiece limit is strict: 101 characters become one [UNK], 100 do not.
    func testTokenizerEnforcesWordLengthBoundaryOffline() throws {
        let tokenizer = try makeOfflineTokenizer()
        // oracle row 17: 101 > MAX_WORD_CHARS=100, so the whole word is exactly one [UNK]=100.
        let overlongWord = String(repeating: "a", count: MAX_WORD_CHARS + 1)
        XCTAssertEqual(overlongWord.count, 101)
        XCTAssertEqual(
            tokenizer.encode(overlongWord),
            [CLS_ID, UNK_ID, SEP_ID],
            "a > MAX_WORD_CHARS word must collapse to a single [UNK]"
        )

        // oracle row 27: exactly 100 characters still take the WordPiece path. Both "a" and
        // "##a" are in the tracked vocab, so no position needs the [UNK] fallback.
        let limitWord = String(repeating: "a", count: MAX_WORD_CHARS)
        let limitTokens = tokenizer.encode(limitWord)
        XCTAssertNotEqual(
            limitTokens,
            [CLS_ID, UNK_ID, SEP_ID],
            "a 100-character word must not take the length-based [UNK] path"
        )
        XCTAssertEqual(limitTokens.first, CLS_ID)
        XCTAssertEqual(limitTokens.last, SEP_ID)
        XCTAssertFalse(limitTokens.contains(UNK_ID), "known single-letter pieces must not become [UNK]")
    }

    /// Greedy WordPiece matching decomposes an unknown word into known subwords when possible.
    func testTokenizerDecomposesUnknownAsciiWordOffline() throws {
        let tokenizer = try makeOfflineTokenizer()
        // oracle row 18: z=1062, ##zz=13213, ##q=4160. The longer candidates are absent from
        // model/vocab.txt, so "zzzqqq" is z + ##zz + ##q + ##q + ##q, not a whole-word [UNK].
        XCTAssertEqual(
            tokenizer.encode("zzzqqq"),
            [CLS_ID, 1062, 13213, 4160, 4160, 4160, SEP_ID],
            "unknown ASCII must decompose into known WordPiece pieces, not [UNK]"
        )
    }

    /// Basic tokenization isolates CJK ideographs before WordPiece matching.
    func testTokenizerIsolatesCJKCharactersOffline() throws {
        let tokenizer = try makeOfflineTokenizer()
        // oracle row 19: 世=1745 is in model/vocab.txt; 你, 好 and 界 are absent and therefore
        // become one [UNK] each. Isolation preserves their order.
        XCTAssertEqual(
            tokenizer.encode("你好世界"),
            [CLS_ID, UNK_ID, UNK_ID, 1745, UNK_ID, SEP_ID],
            "CJK must be tokenized one character at a time, not as a single unknown word"
        )
    }

    /// Empty text produces only the two special tokens.
    func testTokenizerEncodesEmptyInputOffline() throws {
        let tokenizer = try makeOfflineTokenizer()
        // oracle row 25 / assumption A1: standard BERT special-token construction for zero words.
        XCTAssertEqual(tokenizer.encode(""), [CLS_ID, SEP_ID])
    }

    /// Query encoding prepends the retrieval prefix; document encoding does not.
    func testTokenizerQueryAndDocumentPrefixesMatchReferenceIdsOffline() throws {
        let tokenizer = try makeOfflineTokenizer()
        // oracle row 21: what=2054, is=2003, snow=4586, ##fl=10258, ##ake=13808, ?=1029.
        XCTAssertEqual(
            tokenizer.encode("what is snowflake?"),
            [CLS_ID, 2054, 2003, 4586, 10258, 13808, 1029, SEP_ID],
            "encode must be [CLS] + text pieces + [SEP] with no query prefix"
        )
        // oracle row 20: represent=5050, this=2023, sentence=6251, for=2005, searching=6575,
        // relevant=7882, passages=13768, :=1024, followed by the document pieces above.
        XCTAssertEqual(
            tokenizer.encodeQuery("what is snowflake?"),
            [CLS_ID, 5050, 2023, 6251, 2005, 6575, 7882, 13768, 1024, 2054, 2003, 4586, 10258, 13808, 1029, SEP_ID],
            "encodeQuery must prepend QUERY_PREFIX"
        )
    }

    /// Long input is truncated to 512 ids and still terminated by [SEP].
    func testTokenizerTruncatesLongInputOffline() throws {
        let tokenizer = try makeOfflineTokenizer()
        // oracle rows 22-24 and README.md:52: 1000 single-token words exceed 512, so the output
        // keeps the first 510 text tokens between [CLS] and [SEP].
        let truncatedTokens = tokenizer.encode(fillerText(1000))
        XCTAssertEqual(truncatedTokens.count, MAX_SEQ_LEN)
        XCTAssertEqual(truncatedTokens.first, CLS_ID)
        XCTAssertEqual(truncatedTokens.last, SEP_ID)
    }

    func testFreshContextsFirstCallCompletes() throws {
        let engine = try ArcticEmbedS.fromDir(try modelDirOrSkip())
        let ws = Workspace()
        let ids: [UInt16] = [101, 2023, 2003, 1037, 3231, 102]
        let single = engine.embedTokens(ids, ws: ws)
        assertUnitEmbedding(single, "fresh-context single")
        for i in 0..<200 {
            let par = ParallelContext(nThreads: 2 + (i % 3))
            let multi = engine.embedTokensParallel(ids, par: par)
            assertUnitEmbedding(multi, "fresh context \(i)")
            XCTAssertEqual(single, multi, "fresh context \(i) diverged")
        }
    }
}
