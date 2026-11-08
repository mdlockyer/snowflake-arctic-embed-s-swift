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

    func testTokenizerAndInt8AccuracyAgainstFp32Reference() throws {
        let refURL = testDataDir().appendingPathComponent("reference_test_vectors.json")
        let refData = try Data(contentsOf: refURL)
        let records = try JSONDecoder().decode([RefRecord].self, from: refData)
        XCTAssertEqual(records.count, 12)

        let engine = try ArcticEmbedS.fromDir(modelDir())
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
            let cos = cosineSimilarity(emb, rec.embedding_fp32)
            let preview = String(rec.raw_text.prefix(45))
            print(String(
                format: "Record %2d (%@ | %@): cosine_sim vs FP32 = %.6f",
                idx, rec.kind, preview, cos
            ))
            XCTAssertGreaterThan(cos, 0.9985, "Cosine similarity too low on record \(idx): \(cos)")
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
        XCTAssertGreaterThan(meanCos, 0.9992, "Mean cosine similarity below 0.9992: \(meanCos)")

        // Verify that unpadded ragged sequence packing (`embedBatch` / `forwardRaggedCls`) matches single-sequence inference
        let docTexts = records.filter { $0.kind == "document" }.map { $0.raw_text }
        let batchEmbs = engine.embedBatch(docTexts, isQuery: false, numThreads: 1)
        for (i, text) in docTexts.enumerated() {
            let singleEmb = engine.embedDocument(text, ws: ws)
            let cosRagged = cosineSimilarity(singleEmb, batchEmbs[i])
            XCTAssertGreaterThan(
                cosRagged,
                0.99999,
                "Ragged batch mismatch at doc \(i): cos=\(cosRagged)"
            )
        }
    }

    func testAllSimdBackendsAndAe8RoundtripBitIdentical() throws {
        let dir = modelDir()
        let raw = try QuantizedModelRaw.loadAe8(dir.appendingPathComponent("model.ae8"))
        let tok = try BertTokenizer.fromVocabFile(dir.appendingPathComponent("vocab.txt"))

        let ids = tok.encodeQuery("what is snowflake?")
        let ws = Workspace()

        var scalarOut = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        PackedModel.fromRaw(raw, backend: .scalar).forwardCls(ids, ws: ws, outEmb: &scalarOut)

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
            var out = [Float](repeating: 0.0, count: HIDDEN_SIZE)
            PackedModel.fromRaw(raw, backend: b).forwardCls(ids, ws: ws, outEmb: &out)
            let cos = cosineSimilarity(scalarOut, out)
            let maxDiff = zip(scalarOut, out).map { abs($0 - $1) }.max() ?? 0.0
            print(String(
                format: "Backend %@ vs Scalar: cos=%.8f, max_diff=%.8f",
                b.name(), cos, maxDiff
            ))
            XCTAssertGreaterThan(cos, 0.999, "Backend \(b) diverged from Scalar: cos=\(cos)")
        }

        // Test .ae8 serialization round-trip
        let tmpAe8 = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_arctic_embed_s_\(ProcessInfo.processInfo.processIdentifier).ae8")
        try raw.saveAe8(tmpAe8)
        let loadedRaw = try QuantizedModelRaw.loadAe8(tmpAe8)
        try? FileManager.default.removeItem(at: tmpAe8)

        var detectedOut = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        PackedModel.fromRaw(raw, backend: SimdBackend.detect()).forwardCls(
            ids,
            ws: ws,
            outEmb: &detectedOut
        )

        var loadedOut = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        let packedDetected = PackedModel.fromRaw(loadedRaw, backend: SimdBackend.detect())
        packedDetected.forwardCls(ids, ws: ws, outEmb: &loadedOut)
        XCTAssertEqual(detectedOut, loadedOut, ".ae8 roundtrip must be bit-identical")

        XCTAssertEqual(
            packedDetected.layers[11].qkv.packedW.count,
            0,
            "Layer 11 dead qkv must be pruned"
        )
    }

    func testLongSequenceAccuracyAgainstFp64Reference() throws {
        let dataURL = testDataDir().appendingPathComponent("long_reference_vectors.json")
        let data = try Data(contentsOf: dataURL)
        let records = try JSONDecoder().decode([LongRecord].self, from: data)
        XCTAssertEqual(records.count, 7)

        let engine = try ArcticEmbedS.fromDir(modelDir())
        let ws = Workspace()
        var sum: Float = 0.0
        var minCos: Float = 1.0

        for rec in records {
            let emb = engine.embedTokens(rec.ids, ws: ws)
            let cos = cosineSimilarity(emb, rec.embedding_fp32)
            print(String(format: "%@ (%3d tokens): cosine_sim vs FP64 = %.6f", rec.name, rec.ids.count, cos))
            sum += cos
            minCos = min(minCos, cos)
        }
        let mean = sum / Float(records.count)
        print(String(format: "Long-sequence summary: mean_cos = %.6f, min_cos = %.6f", mean, minCos))
        XCTAssertGreaterThan(minCos, 0.998, "long-sequence min cosine too low: \(minCos)")
        XCTAssertGreaterThan(mean, 0.999, "long-sequence mean cosine too low: \(mean)")
    }

    func testParallelForwardMatchesSingleThread() throws {
        let engine = try ArcticEmbedS.fromDir(modelDir())
        if engine.model.backend != .armSme {
            print("skipped: intra-op parallel path needs the SME backend (have \(engine.model.backend.name()))")
            return
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
                let cos = cosineSimilarity(single, multi)
                let maxDiff = zip(single, multi).map { abs($0 - $1) }.max() ?? 0.0
                print(String(format: "threads=%d len=%3d: cos=%.9f max_diff=%.2e", threads, len, cos, maxDiff))
                XCTAssertGreaterThan(
                    cos,
                    0.99999,
                    "parallel diverged: threads=\(threads) len=\(len) cos=\(cos)"
                )
            }
        }
    }

    func testFreshContextsFirstCallCompletes() throws {
        let engine = try ArcticEmbedS.fromDir(modelDir())
        let ws = Workspace()
        let ids: [UInt16] = [101, 2023, 2003, 1037, 3231, 102]
        let single = engine.embedTokens(ids, ws: ws)
        for i in 0..<200 {
            let par = ParallelContext(nThreads: 2 + (i % 3))
            let multi = engine.embedTokensParallel(ids, par: par)
            XCTAssertEqual(single, multi, "fresh context \(i) diverged")
        }
    }

    func testDetMathMatchesLibmClosely() {
        var worstExp = 0.0
        var worstLn = 0.0
        var x = -700.0
        while x < 700.0 {
            let rel = abs((DetMath.expF64(x) - Foundation.exp(x)) / Foundation.exp(x))
            worstExp = max(worstExp, rel)
            x += 0.0137
        }
        var y = 1e-300
        while y < 1e300 {
            let rel = abs((DetMath.lnF64(y) - Foundation.log(y)) / max(abs(Foundation.log(y)), 1e-300))
            worstLn = max(worstLn, rel)
            y *= 1.0137
        }
        XCTAssertLessThan(worstExp, 1e-15, "exp rel err \(worstExp)")
        XCTAssertLessThan(worstLn, 1e-15, "ln rel err \(worstLn)")
        for (b, e) in [(Float(3.7), Float(0.4)), (0.013, 0.6), (250.0, 0.4), (1.0, 0.6)] {
            let ref = Foundation.pow(b, e)
            let rel = abs((DetMath.pow(b, e) - ref) / ref)
            XCTAssertLessThan(rel, 1e-6, "pow(\(b),\(e)) rel err \(rel)")
        }
        XCTAssertEqual(DetMath.pow(0.0, 0.4), 0.0)
    }

    func testCommittedPackMatchesConversionIfCheckpointAvailable() throws {
        let root = repoRoot()
        let defaultWeights = root.appendingPathComponent("weights/model.safetensors")
        let rustWeights = root.deletingLastPathComponent()
            .appendingPathComponent("snowflake-arctic-embed-s-rust/weights/model.safetensors")
        let envSource = ProcessInfo.processInfo.environment["ARCTIC_EMBED_SOURCE_WEIGHTS"]
            .map { URL(fileURLWithPath: $0).appendingPathComponent("model.safetensors") }

        let sourceURL: URL?
        if let env = envSource, FileManager.default.fileExists(atPath: env.path) {
            sourceURL = env
        } else if FileManager.default.fileExists(atPath: defaultWeights.path) {
            sourceURL = defaultWeights
        } else if FileManager.default.fileExists(atPath: rustWeights.path) {
            sourceURL = rustWeights
        } else {
            sourceURL = nil
        }

        guard let source = sourceURL else {
            print("skipped convert test: model.safetensors not found")
            return
        }

        let raw = try QuantizedModelRaw.fromSafetensorsFile(source)
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("convert_test_\(ProcessInfo.processInfo.processIdentifier).ae8")
        try raw.saveAe8(outURL)
        let converted = try Data(contentsOf: outURL)
        try? FileManager.default.removeItem(at: outURL)

        let committed = try Data(contentsOf: root.appendingPathComponent("model/model.ae8"))
        XCTAssertEqual(converted.count, committed.count, "pack size differs from model/model.ae8")
        XCTAssertEqual(converted, committed, "converted model.ae8 differs from committed model/model.ae8")
    }
}
