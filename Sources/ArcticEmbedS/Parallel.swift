import Foundation
import Synchronization
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

private let IDLE_SPIN_NS: UInt64 = 300_000 // 300 µs
private let MAX_STAGES: Int = 128
private let NUM_HEADS_PAIRS: Int = NUM_HEADS / 2
private let FFN_SB_PER_UNIT: Int = 2

@usableFromInline
final class PaddedCounter: @unchecked Sendable {
    private let pad0: (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64) = (0, 0, 0, 0, 0, 0)
    @usableFromInline let value = Atomic<Int>(0)
    private let pad1: (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64) = (0, 0, 0, 0, 0, 0, 0)

    @usableFromInline init() {}
}

@usableFromInline
final class AtomicFlagBox: @unchecked Sendable {
    @usableFromInline let flag = Atomic<Bool>(false)
    @usableFromInline init() {}
}

/// Sense-free generation spin barrier.
@usableFromInline
final class SpinBarrier: @unchecked Sendable {
    @usableFromInline let n: Int
    private let count = Atomic<Int>(0)
    private let generation = Atomic<Int>(0)

    @usableFromInline
    init(n: Int) {
        self.n = n
    }

    @usableFromInline
    @inline(__always)
    func wait() {
        if n <= 1 { return }
        let g = generation.load(ordering: .acquiring)
        let prev = count.wrappingAdd(1, ordering: .acquiringAndReleasing).oldValue
        if prev + 1 == n {
            count.store(0, ordering: .relaxed)
            generation.wrappingAdd(1, ordering: .releasing)
        } else {
            var spins: UInt32 = 0
            while generation.load(ordering: .acquiring) == g {
                spins &+= 1
                if spins < 20_000 {
                    #if arch(arm64)
                    // processor yield hint
                    #endif
                } else if spins < 20_200 {
                    sched_yield()
                } else {
                    usleep(20)
                }
            }
        }
    }
}

@usableFromInline
struct SharedPointers: @unchecked Sendable {
    @usableFromInline var hidden: UnsafeMutablePointer<Float>
    @usableFromInline var tmpHidden: UnsafeMutablePointer<Float>
    @usableFromInline var qkv: UnsafeMutablePointer<Float>
    @usableFromInline var ctx: UnsafeMutablePointer<Float>
    @usableFromInline var inter: UnsafeMutablePointer<Float>
    @usableFromInline var interQ: UnsafeMutablePointer<Int8>
    @usableFromInline var actQ: UnsafeMutablePointer<Int8>
    @usableFromInline var actScale: UnsafeMutablePointer<Float>
    @usableFromInline var qCls: UnsafeMutablePointer<Float>
    @usableFromInline var rowMax: UnsafeMutablePointer<Float>
}

@usableFromInline
struct Job: @unchecked Sendable {
    @usableFromInline var model: Unmanaged<PackedModel>
    @usableFromInline var ids: UnsafePointer<UInt16>
    @usableFromInline var seqLen: Int
    @usableFromInline var shared: SharedPointers
}

@usableFromInline
final class PoolState: @unchecked Sendable {
    @usableFromInline let n: Int
    @usableFromInline let epoch = Atomic<Int>(0)
    @usableFromInline let jobPtr = Atomic<UInt>(0)
    @usableFromInline let done = Atomic<Int>(0)
    @usableFromInline let shutdown = Atomic<Bool>(false)
    @usableFromInline let sleeping: [AtomicFlagBox]
    @usableFromInline let semaphores: [DispatchSemaphore]
    @usableFromInline let exitSemaphores: [DispatchSemaphore]
    @usableFromInline let barrier: SpinBarrier
    @usableFromInline let stageCtr: [PaddedCounter]

    @usableFromInline
    init(n: Int) {
        self.n = n
        self.sleeping = (0..<n).map { _ in AtomicFlagBox() }
        self.semaphores = (0..<n).map { _ in DispatchSemaphore(value: 0) }
        self.exitSemaphores = (0..<n).map { _ in DispatchSemaphore(value: 0) }
        self.barrier = SpinBarrier(n: n)
        self.stageCtr = (0..<(MAX_STAGES * n)).map { _ in PaddedCounter() }
    }
}

/// Reusable context for low-latency single-sequence inference on several cores.
/// Create once and reuse; `nThreads = 2` targets the two P-cluster SME units of M4 Pro/Max.
/// With a non-SME backend it falls back to the single-threaded forward pass.
public final class ParallelContext: @unchecked Sendable {
    public let ws: Workspace
    private let rowMax: AlignedBuf<Float>
    private let state: PoolState

    public init(nThreads: Int) {
        let n = min(max(nThreads, 1), 8)
        let st = PoolState(n: n)
        self.ws = Workspace()
        self.rowMax = AlignedBuf<Float>(zeroed: (INTERMEDIATE_SIZE / 64) * MAX_POSITIONS)
        self.state = st

        if n > 1 {
            for tid in 1..<n {
                let thread = Thread {
                    helperMain(tid: tid, st: st, startEpoch: 0)
                }
                thread.name = "arctic-embed-par-\(tid)"
                thread.qualityOfService = .userInteractive
                thread.start()
            }
        }
    }

    /// Recommended context for `engine`: 2 threads (one per P-cluster SME unit) when the SME
    /// backend is active, otherwise 1 (plain single-threaded forward pass, no helper threads).
    public static func recommended(_ engine: ArcticEmbedS) -> ParallelContext {
        ParallelContext(nThreads: engine.model.backend == .armSme ? 2 : 1)
    }

    public func numThreads() -> Int {
        state.n
    }

    public func forwardCls(
        model: PackedModel,
        inputIds: [UInt16],
        outEmb: inout [Float]
    ) {
        if outEmb.count < HIDDEN_SIZE {
            outEmb = [Float](repeating: 0.0, count: HIDDEN_SIZE)
        }
        inputIds.withUnsafeBufferPointer { idsBuf in
            outEmb.withUnsafeMutableBufferPointer { outBuf in
                forwardCls(model: model, inputIds: idsBuf, outEmb: outBuf.baseAddress!)
            }
        }
    }

    public func forwardCls(
        model: PackedModel,
        inputIds: UnsafeBufferPointer<UInt16>,
        outEmb: UnsafeMutablePointer<Float>
    ) {
        let seqLen = min(inputIds.count, MAX_POSITIONS)
        precondition(seqLen >= 1, "inputIds must not be empty")
        #if arch(arm64)
        if state.n == 1 || model.backend != .armSme {
            model.forwardCls(inputIds: inputIds, ws: ws, outEmb: outEmb)
            return
        }

        let shared = SharedPointers(
            hidden: ws.hidden.ptr,
            tmpHidden: ws.tmpHidden.ptr,
            qkv: ws.qkvBuf.ptr,
            ctx: ws.attnCtx.ptr,
            inter: ws.interBuf.ptr,
            interQ: ws.interQ.ptr,
            actQ: ws.actQ.ptr,
            actScale: ws.actScale.ptr,
            qCls: ws.qCls.ptr,
            rowMax: rowMax.ptr
        )
        var job = Job(
            model: Unmanaged.passUnretained(model),
            ids: inputIds.baseAddress!,
            seqLen: seqLen,
            shared: shared
        )

        let st = state
        st.done.store(0, ordering: .relaxed)
        for c in st.stageCtr {
            c.value.store(0, ordering: .relaxed)
        }

        withUnsafeMutablePointer(to: &job) { jobMutPtr in
            st.jobPtr.store(UInt(bitPattern: jobMutPtr), ordering: .releasing)
            st.epoch.wrappingAdd(1, ordering: .sequentiallyConsistent)
            for tid in 1..<st.n {
                if st.sleeping[tid].flag.exchange(false, ordering: .sequentiallyConsistent) {
                    st.semaphores[tid].signal()
                }
            }

            workerForward(tid: 0, st: st, job: jobMutPtr.pointee, ws: ws)
            tailLastLayer(model: model, ws: ws, seqLen: seqLen, outEmb: outEmb)

            var spins: UInt32 = 0
            while st.done.load(ordering: .acquiring) != st.n - 1 {
                spins &+= 1
                if spins >= 20_000 {
                    sched_yield()
                }
            }
            st.jobPtr.store(0, ordering: .relaxed)
        }
        #else
        model.forwardCls(inputIds: inputIds, ws: ws, outEmb: outEmb)
        #endif
    }

    deinit {
        if state.n > 1 {
            state.shutdown.store(true, ordering: .sequentiallyConsistent)
            state.epoch.wrappingAdd(1, ordering: .sequentiallyConsistent)
            for tid in 1..<state.n {
                if state.sleeping[tid].flag.exchange(false, ordering: .sequentiallyConsistent) {
                    state.semaphores[tid].signal()
                }
            }
            for tid in 1..<state.n {
                state.exitSemaphores[tid].wait()
            }
        }
    }
}

private func setThreadQosUserInteractive() {
    #if canImport(Darwin)
    let QOS_CLASS_USER_INTERACTIVE = qos_class_t(rawValue: 0x21)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    #endif
}

@inline(__always)
private func monotonicNanoseconds() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

private func helperMain(tid: Int, st: PoolState, startEpoch: Int) {
    setThreadQosUserInteractive()
    let helperWs = Workspace()
    var seen = startEpoch

    while true {
        var idleSince = monotonicNanoseconds()
        var spins: UInt32 = 0
        while true {
            let e = st.epoch.load(ordering: .acquiring)
            if e != seen {
                seen = e
                break
            }
            spins &+= 1
            if (spins & 1023) != 0 {
                continue
            }
            if monotonicNanoseconds() &- idleSince < IDLE_SPIN_NS {
                continue
            }
            st.sleeping[tid].flag.store(true, ordering: .sequentiallyConsistent)
            if st.epoch.load(ordering: .sequentiallyConsistent) == seen {
                st.semaphores[tid].wait()
            } else {
                if !st.sleeping[tid].flag.exchange(false, ordering: .sequentiallyConsistent) {
                    st.semaphores[tid].wait()
                }
            }
            idleSince = monotonicNanoseconds()
        }

        if st.shutdown.load(ordering: .sequentiallyConsistent) {
            st.exitSemaphores[tid].signal()
            return
        }

        let rawJob = st.jobPtr.load(ordering: .acquiring)
        if rawJob == 0 {
            continue
        }
        #if arch(arm64)
        let jobPtr = UnsafePointer<Job>(bitPattern: rawJob)!
        workerForward(tid: tid, st: st, job: jobPtr.pointee, ws: helperWs)
        #endif
        st.done.wrappingAdd(1, ordering: .releasing)
    }
}

#if arch(arm64)
private struct Sched {
    let st: PoolState
    let tid: Int
    var stage: Int = 0

    @inline(__always)
    mutating func run(units: Int, _ body: (Int) -> Void) {
        let n = st.n
        let baseIdx = stage * n
        stage += 1
        for k in 0..<n {
            let owner = (tid + k) % n
            let lo = (owner * units) / n
            let hi = ((owner + 1) * units) / n
            let ctr = st.stageCtr[baseIdx + owner]
            while true {
                let u = lo + ctr.value.wrappingAdd(1, ordering: .relaxed).oldValue
                if u >= hi {
                    break
                }
                body(u)
            }
        }
    }
}

@inline(__always)
private func rowChunk(m: Int, n: Int) -> Int {
    let d = 4 * n
    return max((m + d - 1) / d, 4)
}

private func workerForward(tid: Int, st: PoolState, job: Job, ws: Workspace) {
    let n = st.n
    let model = job.model.takeUnretainedValue()
    let m = job.seqLen
    let sh = job.shared
    let bar = st.barrier
    var sched = Sched(st: st, tid: tid, stage: 0)
    let H = HIDDEN_SIZE
    let I = INTERMEDIATE_SIZE
    let I_SB = INTERMEDIATE_SIZE / 64
    let rc = rowChunk(m: m, n: n)
    let rowUnits = (m + rc - 1) / rc
    let invSqrtD: Float = 1.0 / Float(HEAD_DIM).squareRoot()

    var hid = sh.hidden
    var tmp = sh.tmpHidden

    let wordQPtr = model.wordEmbQ.ptr
    let wordSPtr = model.wordEmbScale.ptr
    let posQPtr = model.posEmbQ.ptr
    let posSPtr = model.posEmbScale.ptr
    let type0Ptr = model.type0EmbF32.ptr
    let embLnWPtr = model.embLnWeight.ptr
    let embLnBPtr = model.embLnBias.ptr
    let firstSmooth = model.layers[0].qkv.invSmooth.ptr

    // Embeddings + LN + quantize for layer 0's QKV (rows).
    sched.run(units: rowUnits) { u in
        let r0 = u * rc
        let r1 = min((u + 1) * rc, m)
        for t in r0..<r1 {
            let tok = Int(job.ids[t])
            let hRow = hid.advanced(by: t * H)
            dequantAddEmbeddingsRow(
                wRow: wordQPtr.advanced(by: tok * H),
                wS: wordSPtr[tok],
                pRow: posQPtr.advanced(by: t * H),
                pS: posSPtr[t],
                tRow: type0Ptr,
                hRow: hRow,
                count: H
            )
            sh.actScale[t] = layernormSmoothQuantizeRow(
                xRow: hRow,
                count: H,
                gamma: embLnWPtr,
                beta: embLnBPtr,
                invSmooth: firstSmooth,
                outQ: sh.actQ.advanced(by: t * H),
                eps: LAYER_NORM_EPS
            )
        }
    }
    bar.wait()

    withUnsafeTemporaryAllocation(of: Float.self, capacity: MAX_POSITIONS * 2) { scratchBuf in
        let fullMax = scratchBuf.baseAddress!
        let interScale = fullMax.advanced(by: MAX_POSITIONS)

        for l in 0..<(NUM_LAYERS - 1) {
            let layer = model.layers[l]

            // Unit = head pair: its Q, K, V super-blocks, then attention for those two heads
            smePackActivations(m: m, kIn: H, a: UnsafePointer(sh.actQ), out: ws.actPacked.ptr)
            sched.run(units: NUM_HEADS_PAIRS) { hp in
                for base in [0, NUM_HEADS_PAIRS, 2 * NUM_HEADS_PAIRS] {
                    smeGemmPackedSuperblocks(
                        m: m,
                        aPacked: UnsafePointer(ws.actPacked.ptr),
                        aScale: UnsafePointer(sh.actScale),
                        weights: layer.qkv,
                        residual: nil,
                        out: sh.qkv,
                        sbStart: base + hp,
                        sbCount: 1
                    )
                }
                if m >= 16 {
                    attentionHeadsSme(
                        seqLen: m,
                        qkvBuf: UnsafePointer(sh.qkv),
                        qTF32: ws.smeQT.ptr,
                        kTF32: ws.smeKT.ptr,
                        vContigF32: ws.vHead.ptr,
                        scoresBlk: ws.smeScoresBlk.ptr,
                        pTF32: ws.smePT.ptr,
                        attnCtx: sh.ctx,
                        segRanges: nil,
                        heads: (2 * hp)..<(2 * hp + 2)
                    )
                } else {
                    for h in (2 * hp)..<(2 * hp + 2) {
                        let hOff = h * HEAD_DIM
                        extractKvHead32(
                            qkvBuf: UnsafePointer(sh.qkv),
                            seqLen: m,
                            rowStride: 3 * H,
                            kColOff: H + hOff,
                            vColOff: 2 * H + hOff,
                            kHead: ws.kHead.ptr,
                            vHead: ws.vHead.ptr
                        )
                        for tq in 0..<m {
                            let scores = ws.scores.ptr
                            let q = UnsafePointer(sh.qkv.advanced(by: tq * 3 * H + hOff))
                            computeScoresHead32(
                                qSlice: q,
                                kHead: ws.kHead.ptr,
                                scale: invSqrtD,
                                scores: scores,
                                seqLen: m
                            )
                            softmaxInplace(scores: scores, len: m)
                            let out = sh.ctx.advanced(by: tq * H + hOff)
                            weightedSumHead32(
                                probs: scores,
                                vHead: ws.vHead.ptr,
                                seqLen: m,
                                out32: out
                            )
                        }
                    }
                }
            }
            bar.wait()

            // Quantize attention context (rows).
            let aoSmooth = layer.attnOut.invSmooth.ptr
            sched.run(units: rowUnits) { u in
                let r0 = u * rc
                let r1 = min((u + 1) * rc, m)
                for t in r0..<r1 {
                    sh.actScale[t] = smoothQuantizeRow(
                        xRow: UnsafePointer(sh.ctx.advanced(by: t * H)),
                        count: H,
                        invSmooth: aoSmooth,
                        outQ: sh.actQ.advanced(by: t * H)
                    )
                }
            }
            bar.wait()

            // Attention output projection + residual (super-blocks).
            smePackActivations(m: m, kIn: H, a: UnsafePointer(sh.actQ), out: ws.actPacked.ptr)
            sched.run(units: H / 64) { sb in
                smeGemmPackedSuperblocks(
                    m: m,
                    aPacked: UnsafePointer(ws.actPacked.ptr),
                    aScale: UnsafePointer(sh.actScale),
                    weights: layer.attnOut,
                    residual: UnsafePointer(hid),
                    out: tmp,
                    sbStart: sb,
                    sbCount: 1
                )
            }
            bar.wait()
            swap(&hid, &tmp)

            // LN1 + quantize (rows).
            let ln1W = layer.ln1Weight.ptr
            let ln1B = layer.ln1Bias.ptr
            let ffnInterSmooth = layer.ffnInter.invSmooth.ptr
            sched.run(units: rowUnits) { u in
                let r0 = u * rc
                let r1 = min((u + 1) * rc, m)
                for t in r0..<r1 {
                    sh.actScale[t] = layernormSmoothQuantizeRow(
                        xRow: hid.advanced(by: t * H),
                        count: H,
                        gamma: ln1W,
                        beta: ln1B,
                        invSmooth: ffnInterSmooth,
                        outQ: sh.actQ.advanced(by: t * H),
                        eps: LAYER_NORM_EPS
                    )
                }
            }
            bar.wait()

            // FFN up + GELU pass 1 (unit = 64-column super-block; partial row maxima per super-block).
            let ffnOutSmooth = layer.ffnOut.invSmooth.ptr
            smePackActivations(m: m, kIn: H, a: UnsafePointer(sh.actQ), out: ws.actPacked.ptr)
            sched.run(units: I_SB / FFN_SB_PER_UNIT) { u in
                let sb0 = u * FFN_SB_PER_UNIT
                smeGemmPackedSuperblocks(
                    m: m,
                    aPacked: UnsafePointer(ws.actPacked.ptr),
                    aScale: UnsafePointer(sh.actScale),
                    weights: layer.ffnInter,
                    residual: nil,
                    out: sh.inter,
                    sbStart: sb0,
                    sbCount: FFN_SB_PER_UNIT
                )
                for sb in sb0..<(sb0 + FFN_SB_PER_UNIT) {
                    let rm = sh.rowMax.advanced(by: sb * MAX_POSITIONS)
                    geluSegmentPass1(
                        rows: m,
                        x: sh.inter,
                        rowStride: I,
                        col0: sb * 64,
                        nCols: 64,
                        invSmooth: ffnOutSmooth,
                        rowMax: rm
                    )
                }
            }
            bar.wait()

            // Full-row maxima (every thread computes its own copy), then GELU quantize (super-blocks).
            for r in 0..<m {
                var mx: Float = 0.0
                for sb in 0..<I_SB {
                    let v = sh.rowMax[sb * MAX_POSITIONS + r]
                    if v > mx { mx = v }
                }
                fullMax[r] = mx
                interScale[r] = mx.isFinite ? (max(mx, 1e-8) / 127.0) : (Float.greatestFiniteMagnitude / 127.0)
            }
            sched.run(units: I_SB / FFN_SB_PER_UNIT) { u in
                for sb in (u * FFN_SB_PER_UNIT)..<((u + 1) * FFN_SB_PER_UNIT) {
                    geluSegmentQuantize(
                        rows: m,
                        x: sh.inter,
                        rowStride: I,
                        col0: sb * 64,
                        nCols: 64,
                        rowMaxFull: UnsafePointer(fullMax),
                        q: sh.interQ
                    )
                }
            }
            bar.wait()

            // FFN down + residual (super-blocks).
            smePackActivations(m: m, kIn: I, a: UnsafePointer(sh.interQ), out: ws.interPacked.ptr)
            sched.run(units: H / 64) { sb in
                smeGemmPackedSuperblocks(
                    m: m,
                    aPacked: UnsafePointer(ws.interPacked.ptr),
                    aScale: UnsafePointer(interScale),
                    weights: layer.ffnOut,
                    residual: UnsafePointer(hid),
                    out: tmp,
                    sbStart: sb,
                    sbCount: 1
                )
            }
            bar.wait()
            swap(&hid, &tmp)

            // LN2 + quantize for the next layer's QKV (rows).
            let ln2W = layer.ln2Weight.ptr
            let ln2B = layer.ln2Bias.ptr
            let nextSmooth = model.layers[l + 1].qkv.invSmooth.ptr
            sched.run(units: rowUnits) { u in
                let r0 = u * rc
                let r1 = min((u + 1) * rc, m)
                for t in r0..<r1 {
                    sh.actScale[t] = layernormSmoothQuantizeRow(
                        xRow: hid.advanced(by: t * H),
                        count: H,
                        gamma: ln2W,
                        beta: ln2B,
                        invSmooth: nextSmooth,
                        outQ: sh.actQ.advanced(by: t * H),
                        eps: LAYER_NORM_EPS
                    )
                }
            }
            bar.wait()
        }
    }

    // Layer 11 (unit = head pair): Q for [CLS] only, K/V for all tokens, then [CLS] attention.
    let last = model.layers[NUM_LAYERS - 1]
    let qOnly = last.qOnly!
    let kvOnly = last.kvOnly!
    smePackActivations(m: 1, kIn: H, a: UnsafePointer(sh.actQ), out: ws.interPacked.ptr)
    smePackActivations(m: m, kIn: H, a: UnsafePointer(sh.actQ), out: ws.actPacked.ptr)
    sched.run(units: NUM_HEADS_PAIRS) { hp in
        smeGemmPackedSuperblocks(
            m: 1,
            aPacked: UnsafePointer(ws.interPacked.ptr),
            aScale: UnsafePointer(sh.actScale),
            weights: qOnly,
            residual: nil,
            out: sh.qCls,
            sbStart: hp,
            sbCount: 1
        )
        for base in [0, NUM_HEADS_PAIRS] {
            smeGemmPackedSuperblocks(
                m: m,
                aPacked: UnsafePointer(ws.actPacked.ptr),
                aScale: UnsafePointer(sh.actScale),
                weights: kvOnly,
                residual: nil,
                out: sh.qkv,
                sbStart: base + hp,
                sbCount: 1
            )
        }
        for h in (2 * hp)..<(2 * hp + 2) {
            let hOff = h * HEAD_DIM
            extractKvHead32(
                qkvBuf: UnsafePointer(sh.qkv),
                seqLen: m,
                rowStride: 2 * H,
                kColOff: hOff,
                vColOff: H + hOff,
                kHead: ws.kHead.ptr,
                vHead: ws.vHead.ptr
            )
            let scores = ws.scores.ptr
            computeScoresHead32(
                qSlice: UnsafePointer(sh.qCls.advanced(by: hOff)),
                kHead: ws.kHead.ptr,
                scale: invSqrtD,
                scores: scores,
                seqLen: m
            )
            softmaxInplace(scores: scores, len: m)
            weightedSumHead32(
                probs: scores,
                vHead: ws.vHead.ptr,
                seqLen: m,
                out32: sh.ctx.advanced(by: hOff)
            )
        }
    }
    bar.wait()
}

private func tailLastLayer(
    model: PackedModel,
    ws: Workspace,
    seqLen: Int,
    outEmb: UnsafeMutablePointer<Float>
) {
    let last = model.layers[NUM_LAYERS - 1]
    ws.actScale.ptr[0] = smoothQuantizeRow(
        xRow: ws.attnCtx.ptr,
        count: HIDDEN_SIZE,
        invSmooth: last.attnOut.invSmooth.ptr,
        outQ: ws.actQ.ptr
    )
    gemmInt8Fused(
        m: 1,
        aQ: ws.actQ.ptr,
        aPackedScratch: ws.actPacked.ptr,
        aScale: ws.actScale.ptr,
        weights: last.attnOut,
        residual: ws.hidden.ptr,
        out: ws.tmpHidden.ptr
    )
    ws.actScale.ptr[0] = layernormSmoothQuantizeRow(
        xRow: ws.tmpHidden.ptr,
        count: HIDDEN_SIZE,
        gamma: last.ln1Weight.ptr,
        beta: last.ln1Bias.ptr,
        invSmooth: last.ffnInter.invSmooth.ptr,
        outQ: ws.actQ.ptr,
        eps: LAYER_NORM_EPS
    )
    gemmInt8Fused(
        m: 1,
        aQ: ws.actQ.ptr,
        aPackedScratch: ws.actPacked.ptr,
        aScale: ws.actScale.ptr,
        weights: last.ffnInter,
        residual: nil,
        out: ws.interBuf.ptr
    )
    geluSmoothQuantizeBatch(
        m: 1,
        interBuf: ws.interBuf.ptr,
        invSmooth: last.ffnOut.invSmooth.ptr,
        nCols: INTERMEDIATE_SIZE,
        interQ: ws.interQ.ptr,
        actScale: ws.actScale.ptr,
        useSme: true
    )
    gemmInt8Fused(
        m: 1,
        aQ: ws.interQ.ptr,
        aPackedScratch: ws.interPacked.ptr,
        aScale: ws.actScale.ptr,
        weights: last.ffnOut,
        residual: ws.tmpHidden.ptr,
        out: outEmb
    )
    layernormInplaceRow(
        xRow: outEmb,
        count: HIDDEN_SIZE,
        gamma: last.ln2Weight.ptr,
        beta: last.ln2Bias.ptr,
        eps: LAYER_NORM_EPS
    )
    l2NormalizeInplace(v: outEmb, count: HIDDEN_SIZE)
}
#endif
