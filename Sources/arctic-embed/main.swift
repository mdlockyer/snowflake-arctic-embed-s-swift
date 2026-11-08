import ArcticEmbedS
import Foundation

@inline(never)
private func blackBox(_ x: Float) {
    withExtendedLifetime(x) {}
}

private func truncateStr(_ s: String, maxLen: Int) -> String {
    if s.count <= maxLen {
        return s
    }
    return String(s.prefix(maxLen - 3)) + "..."
}

private func padRight(_ s: String, width: Int) -> String {
    if s.count >= width { return s }
    return s + String(repeating: " ", count: width - s.count)
}

@inline(__always)
private func nowSeconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) * 1e-9
}

func runMain() {
    let args = Array(CommandLine.arguments.dropFirst())
    let quick = args.contains("--quick")
    let modelDir = args.first(where: { !$0.hasPrefix("--") }) ?? defaultModelDir()
    let itersFor: (Int) -> Int = { n in quick ? max(n / 20, 3) : n }
    let warmup = itersFor(10)

    #if os(macOS)
    let hostOS = "macos"
    #elseif os(Linux)
    let hostOS = "linux"
    #else
    let hostOS = "unknown"
    #endif

    #if arch(arm64)
    let hostArch = "aarch64"
    #elseif arch(x86_64)
    let hostArch = "x86_64"
    #else
    let hostArch = "unknown"
    #endif

    print("=========================================================================")
    print("  Snowflake Arctic-Embed-S — Pure-Swift Int8 CPU Inference Engine")
    print("=========================================================================")
    print("Host OS / Arch : \(hostOS) / \(hostArch)")
    let detected = SimdBackend.detect()
    print("Active SIMD ISA: \(detected.name())")

    let modelDirURL = URL(fileURLWithPath: modelDir)
    let t0 = nowSeconds()
    let raw: QuantizedModelRaw
    do {
        raw = try QuantizedModelRaw.loadAe8(modelDirURL.appendingPathComponent(MODEL_FILE))
    } catch {
        fatalError("Failed to load \(modelDir)/model.ae8: \(error)")
    }
    let loadMs = (nowSeconds() - t0) * 1000.0

    let tPack = nowSeconds()
    let packedModel = PackedModel.fromRaw(raw, backend: detected)
    let packMs = (nowSeconds() - tPack) * 1000.0

    let tokenizer: BertTokenizer
    do {
        tokenizer = try BertTokenizer.fromVocabFile(modelDirURL.appendingPathComponent(VOCAB_FILE))
    } catch {
        fatalError("Failed to load vocab.txt: \(error)")
    }

    let engine = ArcticEmbedS(
        tokenizer: tokenizer,
        model: packedModel,
        rawSizeBytes: packedModel.sizeBytes()
    )

    print(String(
        format: "Quantized Size : %.2f MB (3.97x compression vs 133.09 MB FP32)",
        Double(engine.rawSizeBytes) / (1024.0 * 1024.0)
    ))
    print(String(
        format: "Init Latency   : %.2f ms (read model.ae8) + %.2f ms (SIMD pack for %@)",
        loadMs,
        packMs,
        detected.name()
    ))
    print("-------------------------------------------------------------------------")

    let ws = Workspace()

    // 1. Retrieval Demo (Snowflake official queries & documents)
    let queries = [
        "what is snowflake?",
        "Where can I get the best tacos?",
        "How does SIMD Int8 matrix multiplication with ARM64 I8MM SMMLA and AVX2 work?",
    ]
    let documents = [
        "The Data Cloud!",
        "Mexico City of course!",
        "ARMv8.2-A introduces the I8MM extension (SMMLA instruction) which multiplies a 2x8 matrix of 8-bit integers by an 8x2 matrix of 8-bit integers into a 2x2 matrix of 32-bit integers.",
    ]

    let qEmbs = queries.map { engine.embedQuery($0, ws: ws) }
    let dEmbs = documents.map { engine.embedDocument($0, ws: ws) }

    print("Retrieval Similarity Matrix (Query x Document):")
    for (qi, q) in queries.enumerated() {
        var line = "  Q\(qi) (\(padRight(truncateStr(q, maxLen: 42), width: 42))): "
        for di in 0..<documents.count {
            let score = cosineSimilarity(qEmbs[qi], dEmbs[di])
            line += String(format: "D%d=%+.4f  ", di, score)
        }
        print(line)
    }
    print("-------------------------------------------------------------------------")

    // 2. Single-Thread Latency Benchmarks across sequence lengths
    print("Single-Thread Inference Latency (Pure Swift Int8, \(detected.name())):")
    let configs: [(String, Int, Int)] = [
        ("Short   ( 8 tokens)", 8, 500),
        ("Query   (16 tokens)", 16, 400),
        ("Medium  (32 tokens)", 32, 300),
        ("Passage (64 tokens)", 64, 200),
        ("Long   (128 tokens)", 128, 100),
        ("Max    (512 tokens)", 512, 30),
    ]

    let outBuf = AlignedBuf<Float>(zeroed: HIDDEN_SIZE)
    for (label, seqLen, baseIters) in configs {
        let iters = itersFor(baseIters)
        var fakeIds: [UInt16] = [101]
        for i in 0..<(seqLen - 2) {
            fakeIds.append(UInt16(2000 + (i % 1000)))
        }
        fakeIds.append(102)

        fakeIds.withUnsafeBufferPointer { idsBuf in
            for _ in 0..<warmup {
                engine.embedTokensInto(idsBuf, ws: ws, out: outBuf.ptr)
            }
            let start = nowSeconds()
            for _ in 0..<iters {
                engine.embedTokensInto(idsBuf, ws: ws, out: outBuf.ptr)
                blackBox(outBuf.ptr[0])
            }
            let elapsed = nowSeconds() - start
            let usPerCall = (elapsed * 1e6) / Double(iters)
            let seqsPerSec = Double(iters) / elapsed
            let toksPerSec = seqsPerSec * Double(seqLen)
            print(String(
                format: "  %@: %7.1f µs/embed | %7.1f embeds/s | %9.0f tokens/s",
                label, usPerCall, seqsPerSec, toksPerSec
            ))
        }
    }
    print("-------------------------------------------------------------------------")

    // 2.1 Intra-op parallel latency (one sequence on several cores; SME backend only)
    do {
        let par = ParallelContext.recommended(engine)
        if par.numThreads() > 1 {
            print("Intra-Op Parallel Latency (\(par.numThreads()) threads, one sequence):")
            for (label, seqLen, baseIters) in configs {
                let iters = itersFor(baseIters)
                var ids: [UInt16] = [101]
                for i in 0..<(seqLen - 2) {
                    ids.append(UInt16(2000 + (i % 1000)))
                }
                ids.append(102)
                ids.withUnsafeBufferPointer { idsBuf in
                    for _ in 0..<warmup {
                        par.forwardCls(model: engine.model, inputIds: idsBuf, outEmb: outBuf.ptr)
                    }
                    let start = nowSeconds()
                    for _ in 0..<iters {
                        par.forwardCls(model: engine.model, inputIds: idsBuf, outEmb: outBuf.ptr)
                        blackBox(outBuf.ptr[0])
                    }
                    let us = (nowSeconds() - start) * 1e6 / Double(iters)
                    print(String(format: "  %@: %7.1f µs/embed", label, us))
                }
            }
            print("-------------------------------------------------------------------------")
        }
    }

    // 2.5 Per-Operator Breakdown (M = 32, Layer 0)
    do {
        let l0 = engine.model.layers[0]
        let m = 32
        let iters = itersFor(500)

        let t0 = nowSeconds()
        for _ in 0..<iters {
            gemmInt8Fused(
                m: m,
                aQ: ws.actQ.ptr,
                aPackedScratch: ws.actPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: l0.qkv,
                residual: nil,
                out: ws.qkvBuf.ptr
            )
        }
        let qkvUs = (nowSeconds() - t0) * 1e6 / Double(iters)

        let t1 = nowSeconds()
        for _ in 0..<iters {
            gemmInt8Fused(
                m: m,
                aQ: ws.actQ.ptr,
                aPackedScratch: ws.actPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: l0.attnOut,
                residual: ws.hidden.ptr,
                out: ws.tmpHidden.ptr
            )
        }
        let aoUs = (nowSeconds() - t1) * 1e6 / Double(iters)

        let t2 = nowSeconds()
        for _ in 0..<iters {
            gemmInt8Fused(
                m: m,
                aQ: ws.actQ.ptr,
                aPackedScratch: ws.actPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: l0.ffnInter,
                residual: nil,
                out: ws.interBuf.ptr
            )
        }
        let fiUs = (nowSeconds() - t2) * 1e6 / Double(iters)

        let t3 = nowSeconds()
        for _ in 0..<iters {
            gemmInt8Fused(
                m: m,
                aQ: ws.interQ.ptr,
                aPackedScratch: ws.interPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: l0.ffnOut,
                residual: ws.hidden.ptr,
                out: ws.tmpHidden.ptr
            )
        }
        let foUs = (nowSeconds() - t3) * 1e6 / Double(iters)

        let t4 = nowSeconds()
        for _ in 0..<iters {
            for r in 0..<m {
                ws.actScale.ptr[r] = layernormSmoothQuantizeRow(
                    xRow: ws.hidden.ptr.advanced(by: r * 384),
                    count: 384,
                    gamma: l0.ln1Weight.ptr,
                    beta: l0.ln1Bias.ptr,
                    invSmooth: l0.ffnInter.invSmooth.ptr,
                    outQ: ws.actQ.ptr.advanced(by: r * 384),
                    eps: 1e-12
                )
            }
        }
        let lnUs = (nowSeconds() - t4) * 1e6 / Double(iters)

        let t5 = nowSeconds()
        for _ in 0..<iters {
            geluSmoothQuantizeBatch(
                m: m,
                interBuf: ws.interBuf.ptr,
                invSmooth: l0.ffnOut.invSmooth.ptr,
                nCols: 1536,
                interQ: ws.interQ.ptr,
                actScale: ws.actScale.ptr,
                useSme: detected == .armSme
            )
        }
        let geluUs = (nowSeconds() - t5) * 1e6 / Double(iters)

        print(String(
            format: "Layer Breakdown (M=32): QKV=%.1fµs, AttnOut=%.1fµs, FFN_Up=%.1fµs, FFN_Down=%.1fµs, LN+Q=%.1fµs, GELU+Q=%.1fµs",
            qkvUs, aoUs, fiUs, foUs, lnUs, geluUs
        ))
        print("-------------------------------------------------------------------------")
    }

    // 3. Compare SIMD Backends on the same hardware
    print("Backend Comparison on Host CPU (32-token sequence, single thread):")
    var benchIds: [UInt16] = [101]
    for i in 0..<30 {
        benchIds.append(UInt16(2000 + i))
    }
    benchIds.append(102)

    let backends = [
        SimdBackend.armSme,
        SimdBackend.armI8mm,
        SimdBackend.armDotProd,
        SimdBackend.armNeon,
        SimdBackend.x86AvxVnni,
        SimdBackend.x86Avx2,
        SimdBackend.scalar,
    ].filter { $0.isSupported() }

    let refEmb = engine.embedTokens(benchIds, ws: ws)
    for b in backends {
        let pm = PackedModel.fromRaw(raw, backend: b)
        var out = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        var us = 0.0
        benchIds.withUnsafeBufferPointer { idsBuf in
            out.withUnsafeMutableBufferPointer { outSpan in
                for _ in 0..<itersFor(5) {
                    pm.forwardCls(inputIds: idsBuf, ws: ws, outEmb: outSpan.baseAddress!)
                }
                let iters = itersFor(b == .scalar ? 20 : 200)
                let start = nowSeconds()
                for _ in 0..<iters {
                    pm.forwardCls(inputIds: idsBuf, ws: ws, outEmb: outSpan.baseAddress!)
                    blackBox(outSpan[0])
                }
                us = ((nowSeconds() - start) * 1e6) / Double(iters)
            }
        }
        let cos = cosineSimilarity(refEmb, out)
        print(String(
            format: "  %@: %8.1f µs/embed (%6.1f embeds/s) | cos vs top tier = %.7f",
            padRight(b.name(), width: 32),
            us,
            1e6 / us,
            cos
        ))
    }
    print("-------------------------------------------------------------------------")

    // 4. Multi-Threaded Batch Throughput
    let numCpus = max(ProcessInfo.processInfo.activeProcessorCount, 4)
    let batchSize = quick ? 64 : 256
    let sampleDocs: [String] = (0..<batchSize).map { i in
        switch i % 4 {
        case 0:
            return "Snowflake Arctic Embed is a suite of text embedding models optimized for retrieval performance."
        case 1:
            return "That is a happy person enjoying a sunny day in Mexico City with great tacos."
        case 2:
            return "Pure Swift SIMD Int8 inference using ARM64 I8MM SMMLA and x86_64 AVX2 instructions."
        default:
            return "The Data Cloud enables organizations to mobilize their data with near-unlimited scale."
        }
    }

    for threads in [1, 2, 4, 8, numCpus] {
        _ = engine.embedBatch(Array(sampleDocs.prefix(32)), isQuery: false, numThreads: threads)
        let start = nowSeconds()
        let out = engine.embedBatch(sampleDocs, isQuery: false, numThreads: threads)
        blackBox(out[0][0])
        let elapsed = nowSeconds() - start
        print(String(
            format: "Passage Batch (%2d threads, %d docs): %7.1f docs/sec (%.2f ms total)",
            threads, batchSize, Double(batchSize) / elapsed, elapsed * 1000.0
        ))
    }
    print("-------------------------------------------------------------------------")

    let shortDocs: [String] = (0..<batchSize).map { i in
        switch i % 4 {
        case 0: return "The Data Cloud!"
        case 1: return "Mexico City of course!"
        case 2: return "That is a happy person"
        default: return "Today is a sunny day"
        }
    }

    for threads in [1, 2, 4, 8, numCpus] {
        _ = engine.embedBatch(Array(shortDocs.prefix(32)), isQuery: false, numThreads: threads)
        let start = nowSeconds()
        let out = engine.embedBatch(shortDocs, isQuery: false, numThreads: threads)
        blackBox(out[0][0])
        let elapsed = nowSeconds() - start
        print(String(
            format: "Short Ragged  (%2d threads, %d docs): %7.1f docs/sec (%.2f ms total)",
            threads, batchSize, Double(batchSize) / elapsed, elapsed * 1000.0
        ))
    }
    print("=========================================================================")
}

runMain()
