import Foundation

public final class PackedLayer: @unchecked Sendable {
    public let qkv: PackedLinearInt8
    public let qOnly: PackedLinearInt8?
    public let kvOnly: PackedLinearInt8?
    public let attnOut: PackedLinearInt8
    public let ln1Weight: AlignedBuf<Float>
    public let ln1Bias: AlignedBuf<Float>
    public let ffnInter: PackedLinearInt8
    public let ffnOut: PackedLinearInt8
    public let ln2Weight: AlignedBuf<Float>
    public let ln2Bias: AlignedBuf<Float>

    public init(
        qkv: PackedLinearInt8,
        qOnly: PackedLinearInt8?,
        kvOnly: PackedLinearInt8?,
        attnOut: PackedLinearInt8,
        ln1Weight: AlignedBuf<Float>,
        ln1Bias: AlignedBuf<Float>,
        ffnInter: PackedLinearInt8,
        ffnOut: PackedLinearInt8,
        ln2Weight: AlignedBuf<Float>,
        ln2Bias: AlignedBuf<Float>
    ) {
        self.qkv = qkv
        self.qOnly = qOnly
        self.kvOnly = kvOnly
        self.attnOut = attnOut
        self.ln1Weight = ln1Weight
        self.ln1Bias = ln1Bias
        self.ffnInter = ffnInter
        self.ffnOut = ffnOut
        self.ln2Weight = ln2Weight
        self.ln2Bias = ln2Bias
    }
}

public final class PackedModel: @unchecked Sendable {
    public let backend: SimdBackend
    public let wordEmbQ: AlignedBuf<Int8>
    public let wordEmbScale: AlignedBuf<Float>
    public let posEmbQ: AlignedBuf<Int8>
    public let posEmbScale: AlignedBuf<Float>
    /// Pre-dequantized `token_type_embeddings[0]` (`[384]`), added to every token.
    public let type0EmbF32: AlignedBuf<Float>
    public let embLnWeight: AlignedBuf<Float>
    public let embLnBias: AlignedBuf<Float>
    public let layers: [PackedLayer]

    public static func fromRaw(_ raw: QuantizedModelRaw, backend: SimdBackend) -> PackedModel {
        let type0EmbF32 = AlignedBuf<Float>(zeroed: HIDDEN_SIZE)
        let tScale = raw.typeEmbScale[0]
        for j in 0..<HIDDEN_SIZE {
            type0EmbF32[j] = Float(raw.typeEmbQ[j]) * tScale
        }

        let layers = raw.layers.map { l in
            PackedLayer(
                qkv: l.qkv.pack(backend),
                qOnly: l.qOnly?.pack(backend),
                kvOnly: l.kvOnly?.pack(backend),
                attnOut: l.attnOut.pack(backend),
                ln1Weight: AlignedBuf<Float>.fromSlice(l.ln1Weight),
                ln1Bias: AlignedBuf<Float>.fromSlice(l.ln1Bias),
                ffnInter: l.ffnInter.pack(backend),
                ffnOut: l.ffnOut.pack(backend),
                ln2Weight: AlignedBuf<Float>.fromSlice(l.ln2Weight),
                ln2Bias: AlignedBuf<Float>.fromSlice(l.ln2Bias)
            )
        }

        return PackedModel(
            backend: backend,
            wordEmbQ: AlignedBuf<Int8>.fromSlice(raw.wordEmbQ),
            wordEmbScale: AlignedBuf<Float>.fromSlice(raw.wordEmbScale),
            posEmbQ: AlignedBuf<Int8>.fromSlice(raw.posEmbQ),
            posEmbScale: AlignedBuf<Float>.fromSlice(raw.posEmbScale),
            type0EmbF32: type0EmbF32,
            embLnWeight: AlignedBuf<Float>.fromSlice(raw.embLnWeight),
            embLnBias: AlignedBuf<Float>.fromSlice(raw.embLnBias),
            layers: layers
        )
    }

    public init(
        backend: SimdBackend,
        wordEmbQ: AlignedBuf<Int8>,
        wordEmbScale: AlignedBuf<Float>,
        posEmbQ: AlignedBuf<Int8>,
        posEmbScale: AlignedBuf<Float>,
        type0EmbF32: AlignedBuf<Float>,
        embLnWeight: AlignedBuf<Float>,
        embLnBias: AlignedBuf<Float>,
        layers: [PackedLayer]
    ) {
        self.backend = backend
        self.wordEmbQ = wordEmbQ
        self.wordEmbScale = wordEmbScale
        self.posEmbQ = posEmbQ
        self.posEmbScale = posEmbScale
        self.type0EmbF32 = type0EmbF32
        self.embLnWeight = embLnWeight
        self.embLnBias = embLnBias
        self.layers = layers
    }

    /// Total memory footprint of packed weights in bytes.
    public func sizeBytes() -> Int {
        var total = wordEmbQ.count
            + posEmbQ.count
            + (wordEmbScale.count
                + posEmbScale.count
                + type0EmbF32.count
                + embLnWeight.count
                + embLnBias.count) * 4
        for l in layers {
            for lin in [l.qkv, l.attnOut, l.ffnInter, l.ffnOut] {
                total += lin.packedW.count
                    + (lin.wScale.count + lin.bias.count + lin.invSmooth.count + lin.colSumComp.count) * 4
            }
            if let q = l.qOnly, let kv = l.kvOnly {
                for lin in [q, kv] {
                    total += lin.packedW.count
                        + (lin.wScale.count + lin.bias.count + lin.invSmooth.count + lin.colSumComp.count) * 4
                }
            }
            total += (l.ln1Weight.count + l.ln1Bias.count + l.ln2Weight.count + l.ln2Bias.count) * 4
        }
        return total
    }
}

/// Pre-allocated scratchpad for zero-allocation inference up to `MAX_POSITIONS = 512`.
public final class Workspace: @unchecked Sendable {
    public var hidden: AlignedBuf<Float>
    public var tmpHidden: AlignedBuf<Float>
    public let actQ: AlignedBuf<Int8>
    public let actPacked: AlignedBuf<Int8>
    public let actScale: AlignedBuf<Float>
    public let qkvBuf: AlignedBuf<Float>
    public let qCls: AlignedBuf<Float>
    public let attnCtx: AlignedBuf<Float>
    public let kHead: AlignedBuf<Float>
    public let vHead: AlignedBuf<Float>
    public let scores: AlignedBuf<Float>
    public let smeQT: AlignedBuf<Float>
    public let smeKT: AlignedBuf<Float>
    public let smeScoresBlk: AlignedBuf<Float>
    public let smePT: AlignedBuf<Float>
    public let interBuf: AlignedBuf<Float>
    public let interQ: AlignedBuf<Int8>
    public let interPacked: AlignedBuf<Int8>
    public var tokenIds: [UInt16]
    public var segRanges: [(Int, Int)]

    public init() {
        let m = MAX_POSITIONS
        let mPad = m + 64
        self.hidden = AlignedBuf<Float>(zeroed: m * HIDDEN_SIZE)
        self.tmpHidden = AlignedBuf<Float>(zeroed: m * HIDDEN_SIZE)
        self.actQ = AlignedBuf<Int8>(zeroed: m * HIDDEN_SIZE)
        self.actPacked = AlignedBuf<Int8>(zeroed: (m + 16) * HIDDEN_SIZE)
        self.actScale = AlignedBuf<Float>(zeroed: m)
        self.qkvBuf = AlignedBuf<Float>(zeroed: m * 3 * HIDDEN_SIZE)
        self.qCls = AlignedBuf<Float>(zeroed: m * HIDDEN_SIZE)
        self.attnCtx = AlignedBuf<Float>(zeroed: m * HIDDEN_SIZE)
        self.kHead = AlignedBuf<Float>(zeroed: mPad * HEAD_DIM)
        self.vHead = AlignedBuf<Float>(zeroed: mPad * HEAD_DIM)
        self.scores = AlignedBuf<Float>(zeroed: mPad)
        self.smeQT = AlignedBuf<Float>(zeroed: mPad * HEAD_DIM)
        self.smeKT = AlignedBuf<Float>(zeroed: mPad * HEAD_DIM)
        self.smeScoresBlk = AlignedBuf<Float>(zeroed: 16 * mPad)
        self.smePT = AlignedBuf<Float>(zeroed: 16 * mPad)
        self.interBuf = AlignedBuf<Float>(zeroed: m * INTERMEDIATE_SIZE)
        self.interQ = AlignedBuf<Int8>(zeroed: m * INTERMEDIATE_SIZE)
        self.interPacked = AlignedBuf<Int8>(zeroed: (m + 16) * INTERMEDIATE_SIZE)
        self.tokenIds = []
        self.tokenIds.reserveCapacity(m)
        self.segRanges = []
        self.segRanges.reserveCapacity(m)
    }
}

extension PackedModel {
    /// Runs the 12-layer Int8 transformer encoder on `inputIds` and writes the
    /// L2-normalized 384-dim `[CLS]` embedding into `outEmb`.
    public func forwardCls(_ inputIds: [UInt16], ws: Workspace, outEmb: inout [Float]) {
        if outEmb.count < HIDDEN_SIZE {
            outEmb = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        }
        inputIds.withUnsafeBufferPointer { idsBuf in
            outEmb.withUnsafeMutableBufferPointer { outBuf in
                forwardCls(inputIds: idsBuf, ws: ws, outEmb: outBuf.baseAddress!)
            }
        }
    }

    /// Runs the 12-layer Int8 transformer encoder on `inputIds` and writes the
    /// L2-normalized 384-dim `[CLS]` embedding into `outEmb` (must have capacity `>= 384`).
    public func forwardCls(
        inputIds: UnsafeBufferPointer<UInt16>,
        ws: Workspace,
        outEmb: UnsafeMutablePointer<Float>
    ) {
        let seqLen = min(inputIds.count, MAX_POSITIONS)
        precondition(seqLen >= 1, "inputIds must not be empty")
        let idsBase = inputIds.baseAddress!

        // 1. Dequantize Int8 word + position + token_type embeddings & fuse initial LayerNorm + SmoothQuant + Quantize
        let firstInvSmooth = layers[0].qkv.invSmooth.ptr
        let wordQPtr = wordEmbQ.ptr
        let wordSPtr = wordEmbScale.ptr
        let posQPtr = posEmbQ.ptr
        let posSPtr = posEmbScale.ptr
        let type0Ptr = type0EmbF32.ptr
        let embLnWPtr = embLnWeight.ptr
        let embLnBPtr = embLnBias.ptr

        for t in 0..<seqLen {
            let tokIdx = Int(idsBase[t])
            let wOff = tokIdx * HIDDEN_SIZE
            let pOff = t * HIDDEN_SIZE
            let wS = wordSPtr[tokIdx]
            let pS = posSPtr[t]

            let hRow = ws.hidden.ptr.advanced(by: pOff)
            let wRow = wordQPtr.advanced(by: wOff)
            let pRow = posQPtr.advanced(by: pOff)

            dequantAddEmbeddingsRow(
                wRow: wRow,
                wS: wS,
                pRow: pRow,
                pS: pS,
                tRow: type0Ptr,
                hRow: hRow,
                count: HIDDEN_SIZE
            )

            let qRow = ws.actQ.ptr.advanced(by: pOff)
            ws.actScale.ptr[t] = layernormSmoothQuantizeRow(
                xRow: hRow,
                count: HIDDEN_SIZE,
                gamma: embLnWPtr,
                beta: embLnBPtr,
                invSmooth: firstInvSmooth,
                outQ: qRow,
                eps: LAYER_NORM_EPS
            )
        }

        // 2. Layers 0..10 (Full sequence length `seqLen`)
        let invSqrtD: Float = 1.0 / Float(HEAD_DIM).squareRoot()
        let useSme = (backend == .armSme)

        for lIdx in 0..<(NUM_LAYERS - 1) {
            let layer = layers[lIdx]
            let nextQkvSmooth = layers[lIdx + 1].qkv.invSmooth.ptr

            // 2a. Fused QKV Int8 GEMM: [seqLen, 384] x [384, 1152] -> [seqLen, 1152]
            gemmInt8Fused(
                m: seqLen,
                aQ: ws.actQ.ptr,
                aPackedScratch: ws.actPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: layer.qkv,
                residual: nil,
                out: ws.qkvBuf.ptr
            )

            // 2b. Multi-Head Self-Attention (12 heads, head_dim = 32)
            #if arch(arm64)
            if useSme && seqLen >= 16 {
                attentionMultiheadSme(
                    seqLen: seqLen,
                    qkvBuf: ws.qkvBuf.ptr,
                    qTF32: ws.smeQT.ptr,
                    kTF32: ws.smeKT.ptr,
                    vContigF32: ws.vHead.ptr,
                    scoresBlk: ws.smeScoresBlk.ptr,
                    pTF32: ws.smePT.ptr,
                    attnCtx: ws.attnCtx.ptr,
                    segRanges: nil
                )
            } else {
                for h in 0..<NUM_HEADS {
                    let hOff = h * HEAD_DIM
                    extractKvHead32(
                        qkvBuf: ws.qkvBuf.ptr,
                        seqLen: seqLen,
                        rowStride: 3 * HIDDEN_SIZE,
                        kColOff: HIDDEN_SIZE + hOff,
                        vColOff: 2 * HIDDEN_SIZE + hOff,
                        kHead: ws.kHead.ptr,
                        vHead: ws.vHead.ptr
                    )

                    for tq in 0..<seqLen {
                        let qSlice = ws.qkvBuf.ptr.advanced(by: tq * 3 * HIDDEN_SIZE + hOff)
                        let scoresSlice = ws.scores.ptr
                        computeScoresHead32(
                            qSlice: qSlice,
                            kHead: ws.kHead.ptr,
                            scale: invSqrtD,
                            scores: scoresSlice,
                            seqLen: seqLen
                        )
                        softmaxInplace(scores: scoresSlice, len: seqLen)
                        let outHead = ws.attnCtx.ptr.advanced(by: tq * HIDDEN_SIZE + hOff)
                        weightedSumHead32(
                            probs: scoresSlice,
                            vHead: ws.vHead.ptr,
                            seqLen: seqLen,
                            out32: outHead
                        )
                    }
                }
            }
            #else
            for h in 0..<NUM_HEADS {
                let hOff = h * HEAD_DIM
                extractKvHead32(
                    qkvBuf: ws.qkvBuf.ptr,
                    seqLen: seqLen,
                    rowStride: 3 * HIDDEN_SIZE,
                    kColOff: HIDDEN_SIZE + hOff,
                    vColOff: 2 * HIDDEN_SIZE + hOff,
                    kHead: ws.kHead.ptr,
                    vHead: ws.vHead.ptr
                )

                for tq in 0..<seqLen {
                    let qSlice = ws.qkvBuf.ptr.advanced(by: tq * 3 * HIDDEN_SIZE + hOff)
                    let scoresSlice = ws.scores.ptr
                    computeScoresHead32(
                        qSlice: qSlice,
                        kHead: ws.kHead.ptr,
                        scale: invSqrtD,
                        scores: scoresSlice,
                        seqLen: seqLen
                    )
                    softmaxInplace(scores: scoresSlice, len: seqLen)
                    let outHead = ws.attnCtx.ptr.advanced(by: tq * HIDDEN_SIZE + hOff)
                    weightedSumHead32(
                        probs: scoresSlice,
                        vHead: ws.vHead.ptr,
                        seqLen: seqLen,
                        out32: outHead
                    )
                }
            }
            #endif

            // 2c. Quantize attention context + Fused AttnOut GEMM + Residual
            let aoInvSmooth = layer.attnOut.invSmooth.ptr
            for t in 0..<seqLen {
                let off = t * HIDDEN_SIZE
                ws.actScale.ptr[t] = smoothQuantizeRow(
                    xRow: ws.attnCtx.ptr.advanced(by: off),
                    count: HIDDEN_SIZE,
                    invSmooth: aoInvSmooth,
                    outQ: ws.actQ.ptr.advanced(by: off)
                )
            }

            gemmInt8Fused(
                m: seqLen,
                aQ: ws.actQ.ptr,
                aPackedScratch: ws.actPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: layer.attnOut,
                residual: ws.hidden.ptr,
                out: ws.tmpHidden.ptr
            )
            swap(&ws.hidden, &ws.tmpHidden)

            // 2d. Fused LN1 + SmoothQuant + Quantize -> FFN Intermediate GEMM ([seqLen, 1536])
            let ln1W = layer.ln1Weight.ptr
            let ln1B = layer.ln1Bias.ptr
            let ffnInterSmooth = layer.ffnInter.invSmooth.ptr
            for t in 0..<seqLen {
                let off = t * HIDDEN_SIZE
                ws.actScale.ptr[t] = layernormSmoothQuantizeRow(
                    xRow: ws.hidden.ptr.advanced(by: off),
                    count: HIDDEN_SIZE,
                    gamma: ln1W,
                    beta: ln1B,
                    invSmooth: ffnInterSmooth,
                    outQ: ws.actQ.ptr.advanced(by: off),
                    eps: LAYER_NORM_EPS
                )
            }

            gemmInt8Fused(
                m: seqLen,
                aQ: ws.actQ.ptr,
                aPackedScratch: ws.actPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: layer.ffnInter,
                residual: nil,
                out: ws.interBuf.ptr
            )

            // 2e. Fused GELU + SmoothQuant + Quantize -> FFN Output GEMM + Residual -> Fused LN2 + Next Layer Quantize
            geluSmoothQuantizeBatch(
                m: seqLen,
                interBuf: ws.interBuf.ptr,
                invSmooth: layer.ffnOut.invSmooth.ptr,
                nCols: INTERMEDIATE_SIZE,
                interQ: ws.interQ.ptr,
                actScale: ws.actScale.ptr,
                useSme: useSme
            )

            gemmInt8Fused(
                m: seqLen,
                aQ: ws.interQ.ptr,
                aPackedScratch: ws.interPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: layer.ffnOut,
                residual: ws.hidden.ptr,
                out: ws.tmpHidden.ptr
            )
            swap(&ws.hidden, &ws.tmpHidden)

            let ln2W = layer.ln2Weight.ptr
            let ln2B = layer.ln2Bias.ptr
            for t in 0..<seqLen {
                let off = t * HIDDEN_SIZE
                ws.actScale.ptr[t] = layernormSmoothQuantizeRow(
                    xRow: ws.hidden.ptr.advanced(by: off),
                    count: HIDDEN_SIZE,
                    gamma: ln2W,
                    beta: ln2B,
                    invSmooth: nextQkvSmooth,
                    outQ: ws.actQ.ptr.advanced(by: off),
                    eps: LAYER_NORM_EPS
                )
            }
        }

        // 3. Layer 11 (Final Layer): Exact CLS-Token Pruning!
        let lastLayer = layers[NUM_LAYERS - 1]
        let qOnly = lastLayer.qOnly!
        let kvOnly = lastLayer.kvOnly!

        // 3a. Compute Q only for token 0 (M = 1), and KV for all tokens (M = seqLen)
        gemmInt8Fused(
            m: 1,
            aQ: ws.actQ.ptr,
            aPackedScratch: ws.actPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: qOnly,
            residual: nil,
            out: ws.qCls.ptr
        )
        gemmInt8Fused(
            m: seqLen,
            aQ: ws.actQ.ptr,
            aPackedScratch: ws.actPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: kvOnly,
            residual: nil,
            out: ws.qkvBuf.ptr
        )

        // 3b. Vectorized Single-Query Attention (Token 0 attending to 0..seqLen)
        for h in 0..<NUM_HEADS {
            let hOff = h * HEAD_DIM
            let qSlice = ws.qCls.ptr.advanced(by: hOff)
            let scoresSlice = ws.scores.ptr

            extractKvHead32(
                qkvBuf: ws.qkvBuf.ptr,
                seqLen: seqLen,
                rowStride: 2 * HIDDEN_SIZE,
                kColOff: hOff,
                vColOff: HIDDEN_SIZE + hOff,
                kHead: ws.kHead.ptr,
                vHead: ws.vHead.ptr
            )
            computeScoresHead32(
                qSlice: qSlice,
                kHead: ws.kHead.ptr,
                scale: invSqrtD,
                scores: scoresSlice,
                seqLen: seqLen
            )
            softmaxInplace(scores: scoresSlice, len: seqLen)
            let outHead = ws.attnCtx.ptr.advanced(by: hOff)
            weightedSumHead32(
                probs: scoresSlice,
                vHead: ws.vHead.ptr,
                seqLen: seqLen,
                out32: outHead
            )
        }

        // 3c. Token 0 (M = 1) AttnOut + Residual, LN1, FFN Intermediate, GELU, FFN Output + Residual, LN2
        ws.actScale.ptr[0] = smoothQuantizeRow(
            xRow: ws.attnCtx.ptr,
            count: HIDDEN_SIZE,
            invSmooth: lastLayer.attnOut.invSmooth.ptr,
            outQ: ws.actQ.ptr
        )
        gemmInt8Fused(
            m: 1,
            aQ: ws.actQ.ptr,
            aPackedScratch: ws.actPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: lastLayer.attnOut,
            residual: ws.hidden.ptr,
            out: ws.tmpHidden.ptr
        )

        ws.actScale.ptr[0] = layernormSmoothQuantizeRow(
            xRow: ws.tmpHidden.ptr,
            count: HIDDEN_SIZE,
            gamma: lastLayer.ln1Weight.ptr,
            beta: lastLayer.ln1Bias.ptr,
            invSmooth: lastLayer.ffnInter.invSmooth.ptr,
            outQ: ws.actQ.ptr,
            eps: LAYER_NORM_EPS
        )

        gemmInt8Fused(
            m: 1,
            aQ: ws.actQ.ptr,
            aPackedScratch: ws.actPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: lastLayer.ffnInter,
            residual: nil,
            out: ws.interBuf.ptr
        )

        geluSmoothQuantizeBatch(
            m: 1,
            interBuf: ws.interBuf.ptr,
            invSmooth: lastLayer.ffnOut.invSmooth.ptr,
            nCols: INTERMEDIATE_SIZE,
            interQ: ws.interQ.ptr,
            actScale: ws.actScale.ptr,
            useSme: useSme
        )

        gemmInt8Fused(
            m: 1,
            aQ: ws.interQ.ptr,
            aPackedScratch: ws.interPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: lastLayer.ffnOut,
            residual: ws.tmpHidden.ptr,
            out: outEmb
        )

        layernormInplaceRow(
            xRow: outEmb,
            count: HIDDEN_SIZE,
            gamma: lastLayer.ln2Weight.ptr,
            beta: lastLayer.ln2Bias.ptr,
            eps: LAYER_NORM_EPS
        )
        l2NormalizeInplace(v: outEmb, count: HIDDEN_SIZE)
    }

    /// Unpadded Ragged Multi-Sequence Forward Pass ("Flash-VarLen").
    public func forwardRaggedCls(
        sequences: [UnsafeBufferPointer<UInt16>],
        ws: Workspace,
        outEmbs: UnsafeMutablePointer<Float>
    ) {
        let bCount = sequences.count
        if bCount == 0 { return }
        if bCount == 1 {
            forwardCls(inputIds: sequences[0], ws: ws, outEmb: outEmbs)
            return
        }

        ws.segRanges.removeAll(keepingCapacity: true)
        precondition(bCount <= 64, "forwardRaggedCls supports up to 64 packed sequences per call")
        var clsOffsets = [Int](repeating: 0, count: bCount)
        var segBounds = [(Int, Int)](repeating: (0, 0), count: bCount)

        let firstInvSmooth = layers[0].qkv.invSmooth.ptr
        let wordQPtr = wordEmbQ.ptr
        let wordSPtr = wordEmbScale.ptr
        let posQPtr = posEmbQ.ptr
        let posSPtr = posEmbScale.ptr
        let type0Ptr = type0EmbF32.ptr
        let embLnWPtr = embLnWeight.ptr
        let embLnBPtr = embLnBias.ptr

        var cursor = 0

        for b in 0..<bCount {
            let seq = sequences[b]
            let sLen = min(seq.count, MAX_POSITIONS - cursor)
            precondition(sLen >= 1)
            let sStart = cursor
            let sEnd = cursor + sLen
            clsOffsets[b] = sStart
            segBounds[b] = (sStart, sEnd)
            let seqBase = seq.baseAddress!

            for localT in 0..<sLen {
                let globalT = sStart + localT
                ws.segRanges.append((sStart, sEnd))

                let tokIdx = Int(seqBase[localT])
                let wOff = tokIdx * HIDDEN_SIZE
                let pOff = localT * HIDDEN_SIZE
                let dstOff = globalT * HIDDEN_SIZE
                let wS = wordSPtr[tokIdx]
                let pS = posSPtr[localT]

                let hRow = ws.hidden.ptr.advanced(by: dstOff)
                dequantAddEmbeddingsRow(
                    wRow: wordQPtr.advanced(by: wOff),
                    wS: wS,
                    pRow: posQPtr.advanced(by: pOff),
                    pS: pS,
                    tRow: type0Ptr,
                    hRow: hRow,
                    count: HIDDEN_SIZE
                )

                ws.actScale.ptr[globalT] = layernormSmoothQuantizeRow(
                    xRow: hRow,
                    count: HIDDEN_SIZE,
                    gamma: embLnWPtr,
                    beta: embLnBPtr,
                    invSmooth: firstInvSmooth,
                    outQ: ws.actQ.ptr.advanced(by: dstOff),
                    eps: LAYER_NORM_EPS
                )
            }
            cursor = sEnd
        }
        let totalTokens = cursor

        let invSqrtD: Float = 1.0 / Float(HEAD_DIM).squareRoot()
        let useSme = (backend == .armSme)

        for lIdx in 0..<(NUM_LAYERS - 1) {
            let layer = layers[lIdx]
            let nextQkvSmooth = layers[lIdx + 1].qkv.invSmooth.ptr

            gemmInt8Fused(
                m: totalTokens,
                aQ: ws.actQ.ptr,
                aPackedScratch: ws.actPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: layer.qkv,
                residual: nil,
                out: ws.qkvBuf.ptr
            )

            #if arch(arm64)
            for b in 0..<bCount {
                let (sStart, sEnd) = segBounds[b]
                let sLen = sEnd - sStart
                if useSme && sLen >= 16 {
                    // Match forwardCls exactly: SME attention is selected per sequence, never
                    // from the combined pack length. Packing each segment separately also keeps
                    // its V quantization scales independent of every other sequence in the pack.
                    attentionMultiheadSme(
                        seqLen: sLen,
                        qkvBuf: ws.qkvBuf.ptr.advanced(by: sStart * 3 * HIDDEN_SIZE),
                        qTF32: ws.smeQT.ptr,
                        kTF32: ws.smeKT.ptr,
                        vContigF32: ws.vHead.ptr,
                        scoresBlk: ws.smeScoresBlk.ptr,
                        pTF32: ws.smePT.ptr,
                        attnCtx: ws.attnCtx.ptr.advanced(by: sStart * HIDDEN_SIZE),
                        segRanges: nil
                    )
                } else {
                    for h in 0..<NUM_HEADS {
                        let hOff = h * HEAD_DIM
                        extractKvHead32(
                            qkvBuf: ws.qkvBuf.ptr.advanced(by: sStart * 3 * HIDDEN_SIZE),
                            seqLen: sLen,
                            rowStride: 3 * HIDDEN_SIZE,
                            kColOff: HIDDEN_SIZE + hOff,
                            vColOff: 2 * HIDDEN_SIZE + hOff,
                            kHead: ws.kHead.ptr,
                            vHead: ws.vHead.ptr
                        )
                        for ltq in 0..<sLen {
                            let gtq = sStart + ltq
                            let qSlice = ws.qkvBuf.ptr.advanced(by: gtq * 3 * HIDDEN_SIZE + hOff)
                            let scoresSlice = ws.scores.ptr
                            computeScoresHead32(
                                qSlice: qSlice,
                                kHead: ws.kHead.ptr,
                                scale: invSqrtD,
                                scores: scoresSlice,
                                seqLen: sLen
                            )
                            softmaxInplace(scores: scoresSlice, len: sLen)
                            let outHead = ws.attnCtx.ptr.advanced(by: gtq * HIDDEN_SIZE + hOff)
                            weightedSumHead32(
                                probs: scoresSlice,
                                vHead: ws.vHead.ptr,
                                seqLen: sLen,
                                out32: outHead
                            )
                        }
                    }
                }
            }
            #else
            for b in 0..<bCount {
                let (sStart, sEnd) = segBounds[b]
                let sLen = sEnd - sStart
                for h in 0..<NUM_HEADS {
                    let hOff = h * HEAD_DIM
                    extractKvHead32(
                        qkvBuf: ws.qkvBuf.ptr.advanced(by: sStart * 3 * HIDDEN_SIZE),
                        seqLen: sLen,
                        rowStride: 3 * HIDDEN_SIZE,
                        kColOff: HIDDEN_SIZE + hOff,
                        vColOff: 2 * HIDDEN_SIZE + hOff,
                        kHead: ws.kHead.ptr,
                        vHead: ws.vHead.ptr
                    )
                    for ltq in 0..<sLen {
                        let gtq = sStart + ltq
                        let qSlice = ws.qkvBuf.ptr.advanced(by: gtq * 3 * HIDDEN_SIZE + hOff)
                        let scoresSlice = ws.scores.ptr
                        computeScoresHead32(
                            qSlice: qSlice,
                            kHead: ws.kHead.ptr,
                            scale: invSqrtD,
                            scores: scoresSlice,
                            seqLen: sLen
                        )
                        softmaxInplace(scores: scoresSlice, len: sLen)
                        let outHead = ws.attnCtx.ptr.advanced(by: gtq * HIDDEN_SIZE + hOff)
                        weightedSumHead32(
                            probs: scoresSlice,
                            vHead: ws.vHead.ptr,
                            seqLen: sLen,
                            out32: outHead
                        )
                    }
                }
            }
            #endif

            let aoInvSmooth = layer.attnOut.invSmooth.ptr
            for t in 0..<totalTokens {
                let off = t * HIDDEN_SIZE
                ws.actScale.ptr[t] = smoothQuantizeRow(
                    xRow: ws.attnCtx.ptr.advanced(by: off),
                    count: HIDDEN_SIZE,
                    invSmooth: aoInvSmooth,
                    outQ: ws.actQ.ptr.advanced(by: off)
                )
            }

            gemmInt8Fused(
                m: totalTokens,
                aQ: ws.actQ.ptr,
                aPackedScratch: ws.actPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: layer.attnOut,
                residual: ws.hidden.ptr,
                out: ws.tmpHidden.ptr
            )
            swap(&ws.hidden, &ws.tmpHidden)

            let ln1W = layer.ln1Weight.ptr
            let ln1B = layer.ln1Bias.ptr
            let ffnInterSmooth = layer.ffnInter.invSmooth.ptr
            for t in 0..<totalTokens {
                let off = t * HIDDEN_SIZE
                ws.actScale.ptr[t] = layernormSmoothQuantizeRow(
                    xRow: ws.hidden.ptr.advanced(by: off),
                    count: HIDDEN_SIZE,
                    gamma: ln1W,
                    beta: ln1B,
                    invSmooth: ffnInterSmooth,
                    outQ: ws.actQ.ptr.advanced(by: off),
                    eps: LAYER_NORM_EPS
                )
            }

            gemmInt8Fused(
                m: totalTokens,
                aQ: ws.actQ.ptr,
                aPackedScratch: ws.actPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: layer.ffnInter,
                residual: nil,
                out: ws.interBuf.ptr
            )

            geluSmoothQuantizeBatch(
                m: totalTokens,
                interBuf: ws.interBuf.ptr,
                invSmooth: layer.ffnOut.invSmooth.ptr,
                nCols: INTERMEDIATE_SIZE,
                interQ: ws.interQ.ptr,
                actScale: ws.actScale.ptr,
                useSme: useSme
            )

            gemmInt8Fused(
                m: totalTokens,
                aQ: ws.interQ.ptr,
                aPackedScratch: ws.interPacked.ptr,
                aScale: ws.actScale.ptr,
                weights: layer.ffnOut,
                residual: ws.hidden.ptr,
                out: ws.tmpHidden.ptr
            )
            swap(&ws.hidden, &ws.tmpHidden)

            let ln2W = layer.ln2Weight.ptr
            let ln2B = layer.ln2Bias.ptr
            for t in 0..<totalTokens {
                let off = t * HIDDEN_SIZE
                ws.actScale.ptr[t] = layernormSmoothQuantizeRow(
                    xRow: ws.hidden.ptr.advanced(by: off),
                    count: HIDDEN_SIZE,
                    gamma: ln2W,
                    beta: ln2B,
                    invSmooth: nextQkvSmooth,
                    outQ: ws.actQ.ptr.advanced(by: off),
                    eps: LAYER_NORM_EPS
                )
            }
        }

        // 3. Layer 11 (Final Layer): Batched Multi-CLS Token Pruning (`M = bCount`!)
        let lastLayer = layers[NUM_LAYERS - 1]
        let qOnly = lastLayer.qOnly!
        let kvOnly = lastLayer.kvOnly!

        gemmInt8Fused(
            m: totalTokens,
            aQ: ws.actQ.ptr,
            aPackedScratch: ws.actPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: kvOnly,
            residual: nil,
            out: ws.qkvBuf.ptr
        )

        for b in 0..<bCount {
            let clsT = clsOffsets[b]
            if clsT != b {
                ws.actQ.copyWithin(srcRange: (clsT * HIDDEN_SIZE)..<((clsT + 1) * HIDDEN_SIZE), destStart: b * HIDDEN_SIZE)
                ws.actScale.ptr[b] = ws.actScale.ptr[clsT]
                ws.hidden.copyWithin(srcRange: (clsT * HIDDEN_SIZE)..<((clsT + 1) * HIDDEN_SIZE), destStart: b * HIDDEN_SIZE)
            }
        }

        gemmInt8Fused(
            m: bCount,
            aQ: ws.actQ.ptr,
            aPackedScratch: ws.actPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: qOnly,
            residual: nil,
            out: ws.qCls.ptr
        )

        for b in 0..<bCount {
            let (sStart, sEnd) = segBounds[b]
            let sLen = sEnd - sStart
            for h in 0..<NUM_HEADS {
                let hOff = h * HEAD_DIM
                let qSlice = ws.qCls.ptr.advanced(by: b * HIDDEN_SIZE + hOff)
                let scoresSlice = ws.scores.ptr

                extractKvHead32(
                    qkvBuf: ws.qkvBuf.ptr.advanced(by: sStart * 2 * HIDDEN_SIZE),
                    seqLen: sLen,
                    rowStride: 2 * HIDDEN_SIZE,
                    kColOff: hOff,
                    vColOff: HIDDEN_SIZE + hOff,
                    kHead: ws.kHead.ptr,
                    vHead: ws.vHead.ptr
                )
                computeScoresHead32(
                    qSlice: qSlice,
                    kHead: ws.kHead.ptr,
                    scale: invSqrtD,
                    scores: scoresSlice,
                    seqLen: sLen
                )
                softmaxInplace(scores: scoresSlice, len: sLen)
                let outHead = ws.attnCtx.ptr.advanced(by: b * HIDDEN_SIZE + hOff)
                weightedSumHead32(
                    probs: scoresSlice,
                    vHead: ws.vHead.ptr,
                    seqLen: sLen,
                    out32: outHead
                )
            }
        }

        for b in 0..<bCount {
            let off = b * HIDDEN_SIZE
            ws.actScale.ptr[b] = smoothQuantizeRow(
                xRow: ws.attnCtx.ptr.advanced(by: off),
                count: HIDDEN_SIZE,
                invSmooth: lastLayer.attnOut.invSmooth.ptr,
                outQ: ws.actQ.ptr.advanced(by: off)
            )
        }
        gemmInt8Fused(
            m: bCount,
            aQ: ws.actQ.ptr,
            aPackedScratch: ws.actPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: lastLayer.attnOut,
            residual: ws.hidden.ptr,
            out: ws.tmpHidden.ptr
        )

        for b in 0..<bCount {
            let off = b * HIDDEN_SIZE
            ws.actScale.ptr[b] = layernormSmoothQuantizeRow(
                xRow: ws.tmpHidden.ptr.advanced(by: off),
                count: HIDDEN_SIZE,
                gamma: lastLayer.ln1Weight.ptr,
                beta: lastLayer.ln1Bias.ptr,
                invSmooth: lastLayer.ffnInter.invSmooth.ptr,
                outQ: ws.actQ.ptr.advanced(by: off),
                eps: LAYER_NORM_EPS
            )
        }

        gemmInt8Fused(
            m: bCount,
            aQ: ws.actQ.ptr,
            aPackedScratch: ws.actPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: lastLayer.ffnInter,
            residual: nil,
            out: ws.interBuf.ptr
        )

        geluSmoothQuantizeBatch(
            m: bCount,
            interBuf: ws.interBuf.ptr,
            invSmooth: lastLayer.ffnOut.invSmooth.ptr,
            nCols: INTERMEDIATE_SIZE,
            interQ: ws.interQ.ptr,
            actScale: ws.actScale.ptr,
            useSme: useSme
        )

        gemmInt8Fused(
            m: bCount,
            aQ: ws.interQ.ptr,
            aPackedScratch: ws.interPacked.ptr,
            aScale: ws.actScale.ptr,
            weights: lastLayer.ffnOut,
            residual: ws.tmpHidden.ptr,
            out: ws.hidden.ptr
        )

        for b in 0..<bCount {
            let row = ws.hidden.ptr.advanced(by: b * HIDDEN_SIZE)
            layernormInplaceRow(
                xRow: row,
                count: HIDDEN_SIZE,
                gamma: lastLayer.ln2Weight.ptr,
                beta: lastLayer.ln2Bias.ptr,
                eps: LAYER_NORM_EPS
            )
            l2NormalizeInplace(v: row, count: HIDDEN_SIZE)
            UnsafeMutableRawPointer(outEmbs.advanced(by: b * HIDDEN_SIZE)).copyMemory(
                from: UnsafeRawPointer(row),
                byteCount: HIDDEN_SIZE * MemoryLayout<Float>.stride
            )
        }
    }
}
