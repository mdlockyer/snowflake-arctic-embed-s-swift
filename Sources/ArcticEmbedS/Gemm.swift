import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
#if arch(arm64)
import _Builtin_intrinsics.arm.neon
#endif

// MARK: - Low-level NEON Load/Store/Reinterpret Helpers

#if arch(arm64)
@usableFromInline
@inline(__always)
func vld1q_f32(_ p: UnsafePointer<Float>) -> float32x4_t {
    UnsafeRawPointer(p).loadUnaligned(as: float32x4_t.self)
}

@usableFromInline
@inline(__always)
func vst1q_f32(_ p: UnsafeMutablePointer<Float>, _ v: float32x4_t) {
    UnsafeMutableRawPointer(p).storeBytes(of: v, as: float32x4_t.self)
}

@usableFromInline
@inline(__always)
func vld1q_s8(_ p: UnsafePointer<Int8>) -> int8x16_t {
    UnsafeRawPointer(p).loadUnaligned(as: int8x16_t.self)
}

@usableFromInline
@inline(__always)
func vst1q_s8(_ p: UnsafeMutablePointer<Int8>, _ v: int8x16_t) {
    UnsafeMutableRawPointer(p).storeBytes(of: v, as: int8x16_t.self)
}

@usableFromInline
@inline(__always)
func vld1_s8(_ p: UnsafePointer<Int8>) -> int8x8_t {
    UnsafeRawPointer(p).loadUnaligned(as: int8x8_t.self)
}

@usableFromInline
@inline(__always)
func vld1q_s32(_ p: UnsafePointer<Int32>) -> int32x4_t {
    UnsafeRawPointer(p).loadUnaligned(as: int32x4_t.self)
}

@usableFromInline
@inline(__always)
func vst1q_s32(_ p: UnsafeMutablePointer<Int32>, _ v: int32x4_t) {
    UnsafeMutableRawPointer(p).storeBytes(of: v, as: int32x4_t.self)
}

@usableFromInline
@inline(__always)
func vst1q_s16(_ p: UnsafeMutablePointer<Int16>, _ v: int16x8_t) {
    UnsafeMutableRawPointer(p).storeBytes(of: v, as: int16x8_t.self)
}

@usableFromInline
@inline(__always)
func vreinterpretq_f32_s32(_ v: int32x4_t) -> float32x4_t {
    unsafeBitCast(v, to: float32x4_t.self)
}

@usableFromInline
@inline(__always)
func vreinterpretq_s32_f32(_ v: float32x4_t) -> int32x4_t {
    unsafeBitCast(v, to: int32x4_t.self)
}

@usableFromInline
@inline(__always)
func vreinterpretq_s32_s8(_ v: int8x16_t) -> int32x4_t {
    unsafeBitCast(v, to: int32x4_t.self)
}

@usableFromInline
@inline(__always)
func vreinterpretq_s8_s32(_ v: int32x4_t) -> int8x16_t {
    unsafeBitCast(v, to: int8x16_t.self)
}

@usableFromInline
@inline(__always)
func vreinterpretq_s64_s8(_ v: int8x16_t) -> int64x2_t {
    unsafeBitCast(v, to: int64x2_t.self)
}

@usableFromInline
@inline(__always)
func vreinterpretq_s8_s64(_ v: int64x2_t) -> int8x16_t {
    unsafeBitCast(v, to: int8x16_t.self)
}
#endif

// MARK: - SimdBackend

public enum SimdBackend: UInt8, Sendable, Equatable, CaseIterable {
    case armSme = 1
    case armI8mm = 2
    case armDotProd = 3
    case armNeon = 4
    case x86AvxVnni = 5
    case x86Avx2 = 6
    case scalar = 7

    // Rust-style aliases for parity
    public static let ArmSme: SimdBackend = .armSme
    public static let ArmI8mm: SimdBackend = .armI8mm
    public static let ArmDotProd: SimdBackend = .armDotProd
    public static let ArmNeon: SimdBackend = .armNeon
    public static let X86AvxVnni: SimdBackend = .x86AvxVnni
    public static let X86Avx2: SimdBackend = .x86Avx2
    public static let Scalar: SimdBackend = .scalar

    public func isSupported() -> Bool {
        switch self {
        case .scalar:
            return true
        #if arch(arm64)
        case .armNeon:
            return true
        case .armDotProd:
            return detectArmFeature(darwinSysctl: "hw.optional.arm.FEAT_DotProd", linuxHwcap1Bit: 20)
        case .armI8mm:
            return detectArmFeature(darwinSysctl: "hw.optional.arm.FEAT_I8MM", linuxHwcap2Bit: 13)
        case .armSme:
            return detectArmFeature(darwinSysctl: "hw.optional.arm.FEAT_SME2", linuxHwcap2Bit: 37)
        #endif
        #if arch(x86_64)
        case .x86Avx2:
            return (X86AsmKernels.shared.featureMask & 1) != 0
        case .x86AvxVnni:
            return (X86AsmKernels.shared.featureMask & 3) == 3
        #endif
        default:
            return false
        }
    }

    public static func detect() -> SimdBackend {
        #if arch(arm64)
        if SimdBackend.armSme.isSupported() {
            return .armSme
        }
        if SimdBackend.armI8mm.isSupported() {
            return .armI8mm
        }
        if SimdBackend.armDotProd.isSupported() {
            return .armDotProd
        }
        return .armNeon
        #elseif arch(x86_64)
        if SimdBackend.x86AvxVnni.isSupported() {
            return .x86AvxVnni
        }
        if SimdBackend.x86Avx2.isSupported() {
            return .x86Avx2
        }
        return .scalar
        #else
        return .scalar
        #endif
    }

    public func name() -> String {
        switch self {
        case .armSme:
            return "ARM64 FEAT_SME (smopa 16x64)"
        case .armI8mm:
            return "ARM64 FEAT_I8MM (smmla 8x8)"
        case .armDotProd:
            return "ARM64 FEAT_DotProd (sdot 4x8)"
        case .armNeon:
            return "ARM64 NEON (vmull_s8)"
        case .x86AvxVnni:
            return "x86_64 AVX-VNNI (vpdpbusd 4x16)"
        case .x86Avx2:
            return "x86_64 AVX2 (vpmaddubsw 4x16)"
        case .scalar:
            return "Portable Scalar"
        }
    }

    public func toU8() -> UInt8 {
        rawValue
    }

    public static func fromU8(_ v: UInt8) -> SimdBackend? {
        SimdBackend(rawValue: v)
    }
}

#if arch(arm64)
#if canImport(Glibc)
@_silgen_name("getauxval")
private func c_getauxval(_ type: UInt) -> UInt
#endif

@usableFromInline
func detectArmFeature(darwinSysctl: String, linuxHwcap1Bit: UInt64? = nil, linuxHwcap2Bit: UInt64? = nil) -> Bool {
    #if canImport(Darwin)
    var val: Int32 = 0
    var size = MemoryLayout<Int32>.size
    let rc = darwinSysctl.withCString { cStr in
        sysctlbyname(cStr, &val, &size, nil, 0)
    }
    return rc == 0 && val == 1
    #elseif canImport(Glibc)
    if let b1 = linuxHwcap1Bit {
        let AT_HWCAP: UInt = 16
        return (UInt64(c_getauxval(AT_HWCAP)) & (1 << b1)) != 0
    }
    if let b2 = linuxHwcap2Bit {
        let AT_HWCAP2: UInt = 26
        return (UInt64(c_getauxval(AT_HWCAP2)) & (1 << b2)) != 0
    }
    return false
    #else
    return false
    #endif
}
#endif

// MARK: - PackedLinearInt8

/// Pre-packed Int8 weight matrix `W` of logical shape `[N, K]` (out_features x in_features),
/// together with per-output-channel dequantization scales `wScale[N]`, bias `bias[N]`,
/// and SmoothQuant input scaling `invSmooth[K]`.
public final class PackedLinearInt8: @unchecked Sendable {
    public let nOut: Int
    public let kIn: Int
    public let backend: SimdBackend
    public let packedW: AlignedBuf<Int8>
    public let wScale: AlignedBuf<Float>
    public let bias: AlignedBuf<Float>
    public let invSmooth: AlignedBuf<Float>
    public let colSumComp: AlignedBuf<Int32>

    public init(
        nOut: Int,
        kIn: Int,
        backend: SimdBackend,
        packedW: AlignedBuf<Int8>,
        wScale: AlignedBuf<Float>,
        bias: AlignedBuf<Float>,
        invSmooth: AlignedBuf<Float>,
        colSumComp: AlignedBuf<Int32>
    ) {
        self.nOut = nOut
        self.kIn = kIn
        self.backend = backend
        self.packedW = packedW
        self.wScale = wScale
        self.bias = bias
        self.invSmooth = invSmooth
        self.colSumComp = colSumComp
    }

    public static func fromRawRowMajor(
        nOut: Int,
        kIn: Int,
        wRowMajor: [Int8],
        wScale: [Float],
        bias: [Float],
        invSmooth: [Float],
        backend: SimdBackend
    ) -> PackedLinearInt8 {
        if nOut == 0 {
            return PackedLinearInt8(
                nOut: 0,
                kIn: kIn,
                backend: backend,
                packedW: AlignedBuf<Int8>(zeroed: 0),
                wScale: AlignedBuf<Float>(zeroed: 0),
                bias: AlignedBuf<Float>(zeroed: 0),
                invSmooth: AlignedBuf<Float>.fromSlice(invSmooth),
                colSumComp: AlignedBuf<Int32>(zeroed: 0)
            )
        }
        precondition(wRowMajor.count == nOut * kIn)
        precondition(wScale.count == nOut)
        precondition(bias.count == nOut)
        precondition(invSmooth.count == kIn)
        precondition(nOut % 16 == 0, "nOut must be a multiple of 16")
        precondition(kIn % 16 == 0, "kIn must be a multiple of 16")

        let colSumComp: AlignedBuf<Int32>
        if backend == .x86AvxVnni {
            let comp = AlignedBuf<Int32>(zeroed: nOut)
            for n in 0..<nOut {
                let rowOff = n * kIn
                var s: Int32 = 0
                for k in 0..<kIn {
                    s &+= Int32(wRowMajor[rowOff + k])
                }
                comp[n] = -128 &* s
            }
            colSumComp = comp
        } else {
            colSumComp = AlignedBuf<Int32>(zeroed: 0)
        }

        let packedW: AlignedBuf<Int8>
        switch backend {
        case .armSme:
            packedW = packWeightsSme(nOut: nOut, kIn: kIn, w: wRowMajor)
        case .armI8mm:
            packedW = packWeightsI8mm(nOut: nOut, kIn: kIn, w: wRowMajor)
        case .armDotProd:
            packedW = packWeightsSdot(nOut: nOut, kIn: kIn, w: wRowMajor)
        case .x86AvxVnni, .x86Avx2:
            packedW = packWeightsX86(nOut: nOut, kIn: kIn, w: wRowMajor)
        case .armNeon, .scalar:
            packedW = AlignedBuf<Int8>.fromSlice(wRowMajor)
        }

        return PackedLinearInt8(
            nOut: nOut,
            kIn: kIn,
            backend: backend,
            packedW: packedW,
            wScale: AlignedBuf<Float>.fromSlice(wScale),
            bias: AlignedBuf<Float>.fromSlice(bias),
            invSmooth: AlignedBuf<Float>.fromSlice(invSmooth),
            colSumComp: colSumComp
        )
    }
}

// MARK: - Weight Packing

/// Packs `[N, K]` into `[N / 64, K / 4, 4, 64]` for ARM64 `smopa` (`FEAT_SME`).
public func packWeightsSme(nOut: Int, kIn: Int, w: [Int8]) -> AlignedBuf<Int8> {
    precondition(nOut % 64 == 0, "SME requires nOut % 64 == 0")
    let out = AlignedBuf<Int8>(zeroed: nOut * kIn)
    let nSuper = nOut / 64
    let kSteps = kIn / 4

    #if arch(arm64)
    if kSteps % 4 == 0 {
        w.withUnsafeBufferPointer { wBuf in
            let wI32 = UnsafeRawPointer(wBuf.baseAddress!).assumingMemoryBound(to: Int32.self)
            let outI32 = UnsafeMutableRawPointer(out.ptr).assumingMemoryBound(to: Int32.self)
            for nb in 0..<nSuper {
                let colBase = nb * 64
                let nbDst = outI32.advanced(by: nb * kSteps * 64)
                for t in 0..<4 {
                    let tileCol = colBase + t * 16
                    var ks = 0
                    while ks < kSteps {
                        let dstKs = nbDst.advanced(by: ks * 64 + t * 16)
                        for rg in 0..<4 {
                            let c0 = rg * 4
                            let rPtr = wI32.advanced(by: (tileCol + c0) * kSteps + ks)
                            let v0 = vld1q_s32(rPtr)
                            let v1 = vld1q_s32(rPtr.advanced(by: kSteps))
                            let v2 = vld1q_s32(rPtr.advanced(by: 2 * kSteps))
                            let v3 = vld1q_s32(rPtr.advanced(by: 3 * kSteps))

                            let t0 = vtrn1q_s32(v0, v1)
                            let t1 = vtrn2q_s32(v0, v1)
                            let t2 = vtrn1q_s32(v2, v3)
                            let t3 = vtrn2q_s32(v2, v3)

                            vst1q_s32(dstKs.advanced(by: c0), vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)))
                            vst1q_s32(dstKs.advanced(by: 64 + c0), vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)))
                            vst1q_s32(dstKs.advanced(by: 128 + c0), vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)))
                            vst1q_s32(dstKs.advanced(by: 192 + c0), vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)))
                        }
                        ks += 4
                    }
                }
            }
        }
        return out
    }
    #endif

    w.withUnsafeBufferPointer { wBuf in
        let wp = wBuf.baseAddress!
        let op = out.ptr
        var dstIdx = 0
        for nb in 0..<nSuper {
            let colBase = nb * 64
            for ks in 0..<kSteps {
                let kBase = ks * 4
                for t in 0..<4 {
                    for c in 0..<16 {
                        let col = colBase + t * 16 + c
                        let srcOff = col * kIn + kBase
                        op[dstIdx] = wp[srcOff]
                        op[dstIdx + 1] = wp[srcOff + 1]
                        op[dstIdx + 2] = wp[srcOff + 2]
                        op[dstIdx + 3] = wp[srcOff + 3]
                        dstIdx += 4
                    }
                }
            }
        }
    }
    return out
}

/// Packs `[N, K]` into `[N / 8, K / 8, 4, 16]` for ARM64 `smmla` (`i8mm`).
public func packWeightsI8mm(nOut: Int, kIn: Int, w: [Int8]) -> AlignedBuf<Int8> {
    let out = AlignedBuf<Int8>(zeroed: nOut * kIn)
    let nPanels = nOut / 8
    let kSteps = kIn / 8

    #if arch(arm64)
    if kSteps % 2 == 0 {
        w.withUnsafeBufferPointer { wBuf in
            let wPtr = wBuf.baseAddress!
            let outPtr = out.ptr
            for nb in 0..<nPanels {
                let colBase = nb * 8
                let panelDst = outPtr.advanced(by: nb * kSteps * 64)
                var ks = 0
                while ks < kSteps {
                    let kb = ks * 8
                    let dst0 = panelDst.advanced(by: ks * 64)
                    let dst1 = dst0.advanced(by: 64)
                    for p in 0..<4 {
                        let c0 = colBase + 2 * p
                        let v0 = vreinterpretq_s64_s8(vld1q_s8(wPtr.advanced(by: c0 * kIn + kb)))
                        let v1 = vreinterpretq_s64_s8(vld1q_s8(wPtr.advanced(by: (c0 + 1) * kIn + kb)))
                        let lo = vreinterpretq_s8_s64(vtrn1q_s64(v0, v1))
                        let hi = vreinterpretq_s8_s64(vtrn2q_s64(v0, v1))
                        vst1q_s8(dst0.advanced(by: p * 16), lo)
                        vst1q_s8(dst1.advanced(by: p * 16), hi)
                    }
                    ks += 2
                }
            }
        }
        return out
    }
    #endif

    w.withUnsafeBufferPointer { wBuf in
        let wp = wBuf.baseAddress!
        let op = out.ptr
        var dstIdx = 0
        for nb in 0..<nPanels {
            let colBase = nb * 8
            for ks in 0..<kSteps {
                let kBase = ks * 8
                for p in 0..<4 {
                    let c0 = colBase + 2 * p
                    let c1 = c0 + 1
                    for j in 0..<8 {
                        op[dstIdx + j] = wp[c0 * kIn + kBase + j]
                        op[dstIdx + 8 + j] = wp[c1 * kIn + kBase + j]
                    }
                    dstIdx += 16
                }
            }
        }
    }
    return out
}

/// Packs `[N, K]` into `[N / 8, K / 4, 2, 16]` for ARM64 `sdot` (`dotprod`).
public func packWeightsSdot(nOut: Int, kIn: Int, w: [Int8]) -> AlignedBuf<Int8> {
    let out = AlignedBuf<Int8>(zeroed: nOut * kIn)
    let nPanels = nOut / 8
    let kSteps = kIn / 4

    #if arch(arm64)
    if kSteps % 4 == 0 {
        w.withUnsafeBufferPointer { wBuf in
            let wI32 = UnsafeRawPointer(wBuf.baseAddress!).assumingMemoryBound(to: Int32.self)
            let outI32 = UnsafeMutableRawPointer(out.ptr).assumingMemoryBound(to: Int32.self)
            for nb in 0..<nPanels {
                let colBase = nb * 8
                let panelDst = outI32.advanced(by: nb * kSteps * 8)
                var ks = 0
                while ks < kSteps {
                    let dstKs = panelDst.advanced(by: ks * 8)
                    for g in 0..<2 {
                        let c0 = colBase + g * 4
                        let rPtr = wI32.advanced(by: c0 * kSteps + ks)
                        let v0 = vld1q_s32(rPtr)
                        let v1 = vld1q_s32(rPtr.advanced(by: kSteps))
                        let v2 = vld1q_s32(rPtr.advanced(by: 2 * kSteps))
                        let v3 = vld1q_s32(rPtr.advanced(by: 3 * kSteps))

                        let t0 = vtrn1q_s32(v0, v1)
                        let t1 = vtrn2q_s32(v0, v1)
                        let t2 = vtrn1q_s32(v2, v3)
                        let t3 = vtrn2q_s32(v2, v3)

                        vst1q_s32(dstKs.advanced(by: g * 4), vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)))
                        vst1q_s32(dstKs.advanced(by: 8 + g * 4), vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)))
                        vst1q_s32(dstKs.advanced(by: 16 + g * 4), vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)))
                        vst1q_s32(dstKs.advanced(by: 24 + g * 4), vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)))
                    }
                    ks += 4
                }
            }
        }
        return out
    }
    #endif

    w.withUnsafeBufferPointer { wBuf in
        let wp = wBuf.baseAddress!
        let op = out.ptr
        var dstIdx = 0
        for nb in 0..<nPanels {
            let colBase = nb * 8
            for ks in 0..<kSteps {
                let kBase = ks * 4
                for g in 0..<2 {
                    for c in 0..<4 {
                        let col = colBase + g * 4 + c
                        let srcOff = col * kIn + kBase
                        op[dstIdx] = wp[srcOff]
                        op[dstIdx + 1] = wp[srcOff + 1]
                        op[dstIdx + 2] = wp[srcOff + 2]
                        op[dstIdx + 3] = wp[srcOff + 3]
                        dstIdx += 4
                    }
                }
            }
        }
    }
    return out
}

/// Packs `[N, K]` into `[N / 16, K / 4, 2, 32]` for x86_64 AVX2 / AVX-VNNI.
public func packWeightsX86(nOut: Int, kIn: Int, w: [Int8]) -> AlignedBuf<Int8> {
    let out = AlignedBuf<Int8>(zeroed: nOut * kIn)
    let nPanels = nOut / 16
    let kSteps = kIn / 4
    w.withUnsafeBufferPointer { wBuf in
        let wp = wBuf.baseAddress!
        let op = out.ptr
        var dstIdx = 0
        for nb in 0..<nPanels {
            let colBase = nb * 16
            for ks in 0..<kSteps {
                let kBase = ks * 4
                for g in 0..<2 {
                    for c in 0..<8 {
                        let col = colBase + g * 8 + c
                        let srcOff = col * kIn + kBase
                        op[dstIdx] = wp[srcOff]
                        op[dstIdx + 1] = wp[srcOff + 1]
                        op[dstIdx + 2] = wp[srcOff + 2]
                        op[dstIdx + 3] = wp[srcOff + 3]
                        dstIdx += 4
                    }
                }
            }
        }
    }
    return out
}

// MARK: - Activation Packing

@inlinable
public func packActivationsI8mm(
    m: Int,
    k: Int,
    a: UnsafePointer<Int8>,
    packedA: UnsafeMutablePointer<Int8>
) {
    #if arch(arm64)
    packActivationsI8mmNeon(m: m, k: k, a: a, packedA: packedA)
    #else
    let kSteps = k / 8
    var dst = 0
    var r = 0
    while r + 8 <= m {
        for ks in 0..<kSteps {
            let kb = ks * 8
            for p in 0..<4 {
                let r0 = r + 2 * p
                let r1 = r0 + 1
                for j in 0..<8 {
                    packedA[dst + j] = a[r0 * k + kb + j]
                    packedA[dst + 8 + j] = a[r1 * k + kb + j]
                }
                dst += 16
            }
        }
        r += 8
    }
    if r + 4 <= m {
        for ks in 0..<kSteps {
            let kb = ks * 8
            for p in 0..<2 {
                let r0 = r + 2 * p
                let r1 = r0 + 1
                for j in 0..<8 {
                    packedA[dst + j] = a[r0 * k + kb + j]
                    packedA[dst + 8 + j] = a[r1 * k + kb + j]
                }
                dst += 16
            }
        }
        r += 4
    }
    if r + 2 <= m {
        for ks in 0..<kSteps {
            let kb = ks * 8
            for j in 0..<8 {
                packedA[dst + j] = a[r * k + kb + j]
                packedA[dst + 8 + j] = a[(r + 1) * k + kb + j]
            }
            dst += 16
        }
        r += 2
    }
    if r < m {
        for ks in 0..<kSteps {
            let kb = ks * 8
            for j in 0..<8 {
                let v = a[r * k + kb + j]
                packedA[dst + j] = v
                packedA[dst + 8 + j] = v
            }
            dst += 16
        }
    }
    #endif
}

#if arch(arm64)
@usableFromInline
func packActivationsI8mmNeon(
    m: Int,
    k: Int,
    a: UnsafePointer<Int8>,
    packedA: UnsafeMutablePointer<Int8>
) {
    let kSteps = k / 8
    var dstPtr = packedA
    var r = 0

    while r + 8 <= m {
        let r0 = a.advanced(by: r * k)
        let r1 = a.advanced(by: (r + 1) * k)
        let r2 = a.advanced(by: (r + 2) * k)
        let r3 = a.advanced(by: (r + 3) * k)
        let r4 = a.advanced(by: (r + 4) * k)
        let r5 = a.advanced(by: (r + 5) * k)
        let r6 = a.advanced(by: (r + 6) * k)
        let r7 = a.advanced(by: (r + 7) * k)
        for ks in 0..<kSteps {
            let kb = ks * 8
            vst1q_s8(dstPtr, vcombine_s8(vld1_s8(r0.advanced(by: kb)), vld1_s8(r1.advanced(by: kb))))
            vst1q_s8(dstPtr.advanced(by: 16), vcombine_s8(vld1_s8(r2.advanced(by: kb)), vld1_s8(r3.advanced(by: kb))))
            vst1q_s8(dstPtr.advanced(by: 32), vcombine_s8(vld1_s8(r4.advanced(by: kb)), vld1_s8(r5.advanced(by: kb))))
            vst1q_s8(dstPtr.advanced(by: 48), vcombine_s8(vld1_s8(r6.advanced(by: kb)), vld1_s8(r7.advanced(by: kb))))
            dstPtr = dstPtr.advanced(by: 64)
        }
        r += 8
    }

    if r + 4 <= m {
        let r0 = a.advanced(by: r * k)
        let r1 = a.advanced(by: (r + 1) * k)
        let r2 = a.advanced(by: (r + 2) * k)
        let r3 = a.advanced(by: (r + 3) * k)
        for ks in 0..<kSteps {
            let kb = ks * 8
            vst1q_s8(dstPtr, vcombine_s8(vld1_s8(r0.advanced(by: kb)), vld1_s8(r1.advanced(by: kb))))
            vst1q_s8(dstPtr.advanced(by: 16), vcombine_s8(vld1_s8(r2.advanced(by: kb)), vld1_s8(r3.advanced(by: kb))))
            dstPtr = dstPtr.advanced(by: 32)
        }
        r += 4
    }

    if r + 2 <= m {
        let r0 = a.advanced(by: r * k)
        let r1 = a.advanced(by: (r + 1) * k)
        for ks in 0..<kSteps {
            let kb = ks * 8
            vst1q_s8(dstPtr, vcombine_s8(vld1_s8(r0.advanced(by: kb)), vld1_s8(r1.advanced(by: kb))))
            dstPtr = dstPtr.advanced(by: 16)
        }
        r += 2
    }

    if r < m {
        let r0 = a.advanced(by: r * k)
        for ks in 0..<kSteps {
            let kb = ks * 8
            let v = vld1_s8(r0.advanced(by: kb))
            vst1q_s8(dstPtr, vcombine_s8(v, v))
            dstPtr = dstPtr.advanced(by: 16)
        }
    }
}

@usableFromInline
func packActivationsSmeNeon(
    m: Int,
    kIn: Int,
    a: UnsafePointer<Int8>,
    out: UnsafeMutablePointer<Int8>
) {
    let mBlocks = (m + 15) / 16
    let kSteps = kIn / 4
    let aI32 = UnsafeRawPointer(a).assumingMemoryBound(to: Int32.self)
    let outI32 = UnsafeMutableRawPointer(out).assumingMemoryBound(to: Int32.self)
    let zeroV = vdupq_n_s32(0)

    for rb in 0..<mBlocks {
        let rBase = rb * 16
        let validR = min(m - rBase, 16)
        let blockDst = outI32.advanced(by: rb * kSteps * 16)

        if validR == 16 {
            var ks = 0
            while ks + 4 <= kSteps {
                let dstK = blockDst.advanced(by: ks * 16)
                for rg in 0..<4 {
                    let row0 = rg * 4
                    let baseR = (rBase + row0) * kSteps + ks
                    let v0 = vld1q_s32(aI32.advanced(by: baseR))
                    let v1 = vld1q_s32(aI32.advanced(by: baseR + kSteps))
                    let v2 = vld1q_s32(aI32.advanced(by: baseR + 2 * kSteps))
                    let v3 = vld1q_s32(aI32.advanced(by: baseR + 3 * kSteps))

                    let t0 = vtrn1q_s32(v0, v1)
                    let t1 = vtrn2q_s32(v0, v1)
                    let t2 = vtrn1q_s32(v2, v3)
                    let t3 = vtrn2q_s32(v2, v3)

                    vst1q_s32(dstK.advanced(by: row0), vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)))
                    vst1q_s32(dstK.advanced(by: 16 + row0), vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)))
                    vst1q_s32(dstK.advanced(by: 32 + row0), vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)))
                    vst1q_s32(dstK.advanced(by: 48 + row0), vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)))
                }
                ks += 4
            }
        } else {
            var ks = 0
            while ks + 4 <= kSteps {
                let dstK = blockDst.advanced(by: ks * 16)
                for rg in 0..<4 {
                    let row0 = rg * 4
                    if row0 >= validR {
                        vst1q_s32(dstK.advanced(by: row0), zeroV)
                        vst1q_s32(dstK.advanced(by: 16 + row0), zeroV)
                        vst1q_s32(dstK.advanced(by: 32 + row0), zeroV)
                        vst1q_s32(dstK.advanced(by: 48 + row0), zeroV)
                        continue
                    }
                    let v0 = vld1q_s32(aI32.advanced(by: (rBase + row0) * kSteps + ks))
                    let v1 = (row0 + 1 < validR)
                        ? vld1q_s32(aI32.advanced(by: (rBase + row0 + 1) * kSteps + ks))
                        : zeroV
                    let v2 = (row0 + 2 < validR)
                        ? vld1q_s32(aI32.advanced(by: (rBase + row0 + 2) * kSteps + ks))
                        : zeroV
                    let v3 = (row0 + 3 < validR)
                        ? vld1q_s32(aI32.advanced(by: (rBase + row0 + 3) * kSteps + ks))
                        : zeroV

                    let t0 = vtrn1q_s32(v0, v1)
                    let t1 = vtrn2q_s32(v0, v1)
                    let t2 = vtrn1q_s32(v2, v3)
                    let t3 = vtrn2q_s32(v2, v3)

                    vst1q_s32(dstK.advanced(by: row0), vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)))
                    vst1q_s32(dstK.advanced(by: 16 + row0), vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)))
                    vst1q_s32(dstK.advanced(by: 32 + row0), vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)))
                    vst1q_s32(dstK.advanced(by: 48 + row0), vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)))
                }
                ks += 4
            }
        }
    }
}

public func smePackActivations(
    m: Int,
    kIn: Int,
    a: UnsafePointer<Int8>,
    out: UnsafeMutablePointer<Int8>
) {
    packActivationsSmeNeon(m: m, kIn: kIn, a: a, out: out)
}

public func smeGemmPackedSuperblocks(
    m: Int,
    aPacked: UnsafePointer<Int8>,
    aScale: UnsafePointer<Float>,
    weights: PackedLinearInt8,
    residual: UnsafePointer<Float>?,
    out: UnsafeMutablePointer<Float>,
    sbStart: Int,
    sbCount: Int
) {
    if m == 0 || sbCount == 0 { return }
    let nOut = weights.nOut
    let kSteps = weights.kIn / 4
    let col0 = sbStart * 64
    let mBlocks = (m + 15) / 16
    gemmSmeKernel(
        mBlocks: mBlocks,
        m: m,
        nOut: nOut,
        kSteps: kSteps,
        nSuper: sbCount,
        aPacked: aPacked,
        bBase: weights.packedW.ptr.advanced(by: sbStart * kSteps * 256),
        wsBase: weights.wScale.ptr.advanced(by: col0),
        biasBase: weights.bias.ptr.advanced(by: col0),
        aScale: aScale,
        resPtr: residual?.advanced(by: col0),
        outBase: out.advanced(by: col0)
    )
}

@usableFromInline
func gemmSmeKernel(
    mBlocks: Int,
    m: Int,
    nOut: Int,
    kSteps: Int,
    nSuper: Int,
    aPacked: UnsafePointer<Int8>,
    bBase: UnsafePointer<Int8>,
    wsBase: UnsafePointer<Float>,
    biasBase: UnsafePointer<Float>,
    aScale: UnsafePointer<Float>,
    resPtr: UnsafePointer<Float>?,
    outBase: UnsafeMutablePointer<Float>
) {
    let rowblockFn = AArch64AsmKernels.shared.gemmSmeRowblock
    let nBytes = UInt64(nOut * 4)
    let kStepsU = UInt64(kSteps)
    let nSuperU = UInt64(nSuper)

    for rb in 0..<mBlocks {
        let rBase = rb * 16
        let validR = UInt64(min(m - rBase, 16))
        let paBlock = aPacked.advanced(by: rb * kSteps * 64)
        let pasBase = aScale.advanced(by: rBase)
        let poutRb = outBase.advanced(by: rBase * nOut)
        let presRb = resPtr?.advanced(by: rBase * nOut)
        let packedDims = kStepsU | (nBytes << 16) | (validR << 32) | (nSuperU << 48)
        rowblockFn(paBlock, bBase, wsBase, biasBase, pasBase, presRb, poutRb, packedDims)
    }
}
#endif

// MARK: - Fused Int8 GEMM Dispatch

/// Computes `out[m, n] = (sum_k aQ[m, k] * wQ[n, k]) * (aScale[m] * wScale[n]) + bias[n] (+ residual[m, n])`
/// using the optimal SIMD tier on the current CPU.
@inlinable
public func gemmInt8Fused(
    m: Int,
    aQ: UnsafePointer<Int8>,
    aPackedScratch: UnsafeMutablePointer<Int8>,
    aScale: UnsafePointer<Float>,
    weights: PackedLinearInt8,
    residual: UnsafePointer<Float>?,
    out: UnsafeMutablePointer<Float>
) {
    if m == 0 { return }
    switch weights.backend {
    #if arch(arm64)
    case .armSme:
        gemmSmeFused(
            m: m,
            aQ: aQ,
            aPacked: aPackedScratch,
            aScale: aScale,
            weights: weights,
            residual: residual,
            out: out
        )
    case .armI8mm:
        gemmI8mmFused(
            m: m,
            aQ: aQ,
            aPacked: aPackedScratch,
            aScale: aScale,
            weights: weights,
            residual: residual,
            out: out
        )
    case .armDotProd:
        gemmSdotFused(
            m: m,
            aQ: aQ,
            aScale: aScale,
            weights: weights,
            residual: residual,
            out: out
        )
    case .armNeon:
        gemmNeonFused(
            m: m,
            aQ: aQ,
            aScale: aScale,
            weights: weights,
            residual: residual,
            out: out
        )
    #endif
    #if arch(x86_64)
    case .x86AvxVnni:
        gemmX86AvxVnniFused(
            m: m,
            aQ: aQ,
            aU8Scratch: aPackedScratch,
            aScale: aScale,
            weights: weights,
            residual: residual,
            out: out
        )
    case .x86Avx2:
        gemmX86Avx2Fused(
            m: m,
            aQ: aQ,
            aScale: aScale,
            weights: weights,
            residual: residual,
            out: out
        )
    #endif
    default:
        gemmScalarFused(
            m: m,
            aQ: aQ,
            aScale: aScale,
            weights: weights,
            residual: residual,
            out: out
        )
    }
}

#if arch(arm64)
@usableFromInline
func gemmSmeFused(
    m: Int,
    aQ: UnsafePointer<Int8>,
    aPacked: UnsafeMutablePointer<Int8>,
    aScale: UnsafePointer<Float>,
    weights: PackedLinearInt8,
    residual: UnsafePointer<Float>?,
    out: UnsafeMutablePointer<Float>
) {
    let nOut = weights.nOut
    let kIn = weights.kIn
    let kSteps = kIn / 4
    let nSuper = nOut / 64
    let mBlocks = (m + 15) / 16

    packActivationsSmeNeon(m: m, kIn: kIn, a: aQ, out: aPacked)
    gemmSmeKernel(
        mBlocks: mBlocks,
        m: m,
        nOut: nOut,
        kSteps: kSteps,
        nSuper: nSuper,
        aPacked: aPacked,
        bBase: weights.packedW.ptr,
        wsBase: weights.wScale.ptr,
        biasBase: weights.bias.ptr,
        aScale: aScale,
        resPtr: residual,
        outBase: out
    )
}

@usableFromInline
func gemmI8mmFused(
    m: Int,
    aQ: UnsafePointer<Int8>,
    aPacked: UnsafeMutablePointer<Int8>,
    aScale: UnsafePointer<Float>,
    weights: PackedLinearInt8,
    residual: UnsafePointer<Float>?,
    out: UnsafeMutablePointer<Float>
) {
    let nOut = weights.nOut
    let kIn = weights.kIn
    let kSteps = kIn / 8
    let nPanels = nOut / 8

    packActivationsI8mmNeon(m: m, k: kIn, a: aQ, packedA: aPacked)

    let bBase = weights.packedW.ptr
    let wsBase = weights.wScale.ptr
    let biasBase = weights.bias.ptr
    let kernels = AArch64AsmKernels.shared

    var accBuf = (
        SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero,
        SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero,
        SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero,
        SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero
    )

    if m == 1 {
        let sRow = vdupq_n_f32(aScale[0])
        withUnsafeMutablePointer(to: &accBuf) { accTuplePtr in
            let accI32 = UnsafeMutableRawPointer(accTuplePtr).assumingMemoryBound(to: Int32.self)
            let accVecs = UnsafeMutableRawPointer(accTuplePtr).assumingMemoryBound(to: int32x4_t.self)
            var nb = 0
            while nb + 2 <= nPanels {
                let col0 = nb * 8
                let col1 = (nb + 1) * 8
                let pb0 = bBase.advanced(by: nb * kSteps * 64)
                let pb1 = bBase.advanced(by: (nb + 1) * kSteps * 64)

                kernels.smmlaM12Panel(aPacked, pb0, pb1, kSteps, accI32)

                let c0 = accVecs[0]
                let c1 = accVecs[1]
                let c2 = accVecs[2]
                let c3 = accVecs[3]
                let c4 = accVecs[4]
                let c5 = accVecs[5]
                let c6 = accVecs[6]
                let c7 = accVecs[7]

                let acc0_3 = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c0), vget_low_s32(c1)))
                let acc4_7 = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c2), vget_low_s32(c3)))
                let acc8_11 = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c4), vget_low_s32(c5)))
                let acc12_15 = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c6), vget_low_s32(c7)))

                let sc0 = vmulq_f32(sRow, vld1q_f32(wsBase.advanced(by: col0)))
                let sc1 = vmulq_f32(sRow, vld1q_f32(wsBase.advanced(by: col0 + 4)))
                let sc2 = vmulq_f32(sRow, vld1q_f32(wsBase.advanced(by: col1)))
                let sc3 = vmulq_f32(sRow, vld1q_f32(wsBase.advanced(by: col1 + 4)))

                var bias0 = vld1q_f32(biasBase.advanced(by: col0))
                var bias1 = vld1q_f32(biasBase.advanced(by: col0 + 4))
                var bias2 = vld1q_f32(biasBase.advanced(by: col1))
                var bias3 = vld1q_f32(biasBase.advanced(by: col1 + 4))

                if let rp = residual {
                    bias0 = vaddq_f32(bias0, vld1q_f32(rp.advanced(by: col0)))
                    bias1 = vaddq_f32(bias1, vld1q_f32(rp.advanced(by: col0 + 4)))
                    bias2 = vaddq_f32(bias2, vld1q_f32(rp.advanced(by: col1)))
                    bias3 = vaddq_f32(bias3, vld1q_f32(rp.advanced(by: col1 + 4)))
                }

                vst1q_f32(out.advanced(by: col0), vfmaq_f32(bias0, acc0_3, sc0))
                vst1q_f32(out.advanced(by: col0 + 4), vfmaq_f32(bias1, acc4_7, sc1))
                vst1q_f32(out.advanced(by: col1), vfmaq_f32(bias2, acc8_11, sc2))
                vst1q_f32(out.advanced(by: col1 + 4), vfmaq_f32(bias3, acc12_15, sc3))

                nb += 2
            }
        }
        return
    }

    withUnsafeMutablePointer(to: &accBuf) { accTuplePtr in
        let accI32 = UnsafeMutableRawPointer(accTuplePtr).assumingMemoryBound(to: Int32.self)
        let accVecs = UnsafeMutableRawPointer(accTuplePtr).assumingMemoryBound(to: int32x4_t.self)

        for nb in 0..<nPanels {
            let col = nb * 8
            let pbPanel = bBase.advanced(by: nb * kSteps * 64)
            let wsLo = vld1q_f32(wsBase.advanced(by: col))
            let wsHi = vld1q_f32(wsBase.advanced(by: col + 4))
            let biasLo = vld1q_f32(biasBase.advanced(by: col))
            let biasHi = vld1q_f32(biasBase.advanced(by: col + 4))

            var r = 0
            var aTileOffset = 0

            while r + 8 <= m {
                let pa = aPacked.advanced(by: aTileOffset)
                kernels.smmla8x8Tile(pa, pbPanel, kSteps, accI32)

                writePairI8mm(
                    r: r, col: col, nOut: nOut,
                    c0: accVecs[0], c1: accVecs[1], c2: accVecs[2], c3: accVecs[3],
                    aScale: aScale, wsLo: wsLo, wsHi: wsHi, biasLo: biasLo, biasHi: biasHi,
                    resBase: residual, outBase: out
                )
                writePairI8mm(
                    r: r + 2, col: col, nOut: nOut,
                    c0: accVecs[4], c1: accVecs[5], c2: accVecs[6], c3: accVecs[7],
                    aScale: aScale, wsLo: wsLo, wsHi: wsHi, biasLo: biasLo, biasHi: biasHi,
                    resBase: residual, outBase: out
                )
                writePairI8mm(
                    r: r + 4, col: col, nOut: nOut,
                    c0: accVecs[8], c1: accVecs[9], c2: accVecs[10], c3: accVecs[11],
                    aScale: aScale, wsLo: wsLo, wsHi: wsHi, biasLo: biasLo, biasHi: biasHi,
                    resBase: residual, outBase: out
                )
                writePairI8mm(
                    r: r + 6, col: col, nOut: nOut,
                    c0: accVecs[12], c1: accVecs[13], c2: accVecs[14], c3: accVecs[15],
                    aScale: aScale, wsLo: wsLo, wsHi: wsHi, biasLo: biasLo, biasHi: biasHi,
                    resBase: residual, outBase: out
                )

                aTileOffset += kSteps * 64
                r += 8
            }

            if r + 4 <= m {
                let pa = aPacked.advanced(by: aTileOffset)
                kernels.smmla4x8Tile(pa, pbPanel, kSteps, accI32)

                writePairI8mm(
                    r: r, col: col, nOut: nOut,
                    c0: accVecs[0], c1: accVecs[1], c2: accVecs[2], c3: accVecs[3],
                    aScale: aScale, wsLo: wsLo, wsHi: wsHi, biasLo: biasLo, biasHi: biasHi,
                    resBase: residual, outBase: out
                )
                writePairI8mm(
                    r: r + 2, col: col, nOut: nOut,
                    c0: accVecs[4], c1: accVecs[5], c2: accVecs[6], c3: accVecs[7],
                    aScale: aScale, wsLo: wsLo, wsHi: wsHi, biasLo: biasLo, biasHi: biasHi,
                    resBase: residual, outBase: out
                )

                aTileOffset += kSteps * 32
                r += 4
            }

            if r + 2 <= m {
                let pa = aPacked.advanced(by: aTileOffset)
                kernels.smmla2x8Tile(pa, pbPanel, kSteps, accI32)

                writePairI8mm(
                    r: r, col: col, nOut: nOut,
                    c0: accVecs[0], c1: accVecs[1], c2: accVecs[2], c3: accVecs[3],
                    aScale: aScale, wsLo: wsLo, wsHi: wsHi, biasLo: biasLo, biasHi: biasHi,
                    resBase: residual, outBase: out
                )

                aTileOffset += kSteps * 16
                r += 2
            }

            if r < m {
                let pa = aPacked.advanced(by: aTileOffset)
                kernels.smmla2x8Tile(pa, pbPanel, kSteps, accI32)

                let sR0 = vdupq_n_f32(aScale[r])
                let rowOffset = r * nOut
                let wsRLo = vmulq_f32(sR0, wsLo)
                let wsRHi = vmulq_f32(sR0, wsHi)
                var bLo = biasLo
                var bHi = biasHi
                if let rp = residual {
                    bLo = vaddq_f32(bLo, vld1q_f32(rp.advanced(by: rowOffset + col)))
                    bHi = vaddq_f32(bHi, vld1q_f32(rp.advanced(by: rowOffset + col + 4)))
                }
                let r0Lo = vcvtq_f32_s32(vcombine_s32(vget_low_s32(accVecs[0]), vget_low_s32(accVecs[1])))
                let r0Hi = vcvtq_f32_s32(vcombine_s32(vget_low_s32(accVecs[2]), vget_low_s32(accVecs[3])))
                vst1q_f32(out.advanced(by: rowOffset + col), vfmaq_f32(bLo, r0Lo, wsRLo))
                vst1q_f32(out.advanced(by: rowOffset + col + 4), vfmaq_f32(bHi, r0Hi, wsRHi))
            }
        }
    }
}

@usableFromInline
@inline(__always)
func writePairI8mm(
    r: Int,
    col: Int,
    nOut: Int,
    c0: int32x4_t,
    c1: int32x4_t,
    c2: int32x4_t,
    c3: int32x4_t,
    aScale: UnsafePointer<Float>,
    wsLo: float32x4_t,
    wsHi: float32x4_t,
    biasLo: float32x4_t,
    biasHi: float32x4_t,
    resBase: UnsafePointer<Float>?,
    outBase: UnsafeMutablePointer<Float>
) {
    let s0 = vdupq_n_f32(aScale[r])
    let s1 = vdupq_n_f32(aScale[r + 1])

    let r0Lo = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c0), vget_low_s32(c1)))
    let r0Hi = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c2), vget_low_s32(c3)))
    let r1Lo = vcvtq_f32_s32(vcombine_s32(vget_high_s32(c0), vget_high_s32(c1)))
    let r1Hi = vcvtq_f32_s32(vcombine_s32(vget_high_s32(c2), vget_high_s32(c3)))

    let off0 = r * nOut + col
    let off1 = (r + 1) * nOut + col

    let b0Lo: float32x4_t
    let b0Hi: float32x4_t
    let b1Lo: float32x4_t
    let b1Hi: float32x4_t
    if let rp = resBase {
        b0Lo = vaddq_f32(biasLo, vld1q_f32(rp.advanced(by: off0)))
        b0Hi = vaddq_f32(biasHi, vld1q_f32(rp.advanced(by: off0 + 4)))
        b1Lo = vaddq_f32(biasLo, vld1q_f32(rp.advanced(by: off1)))
        b1Hi = vaddq_f32(biasHi, vld1q_f32(rp.advanced(by: off1 + 4)))
    } else {
        b0Lo = biasLo
        b0Hi = biasHi
        b1Lo = biasLo
        b1Hi = biasHi
    }

    vst1q_f32(outBase.advanced(by: off0), vfmaq_f32(b0Lo, r0Lo, vmulq_f32(s0, wsLo)))
    vst1q_f32(outBase.advanced(by: off0 + 4), vfmaq_f32(b0Hi, r0Hi, vmulq_f32(s0, wsHi)))
    vst1q_f32(outBase.advanced(by: off1), vfmaq_f32(b1Lo, r1Lo, vmulq_f32(s1, wsLo)))
    vst1q_f32(outBase.advanced(by: off1 + 4), vfmaq_f32(b1Hi, r1Hi, vmulq_f32(s1, wsHi)))
}

@usableFromInline
func gemmSdotFused(
    m: Int,
    aQ: UnsafePointer<Int8>,
    aScale: UnsafePointer<Float>,
    weights: PackedLinearInt8,
    residual: UnsafePointer<Float>?,
    out: UnsafeMutablePointer<Float>
) {
    let nOut = weights.nOut
    let kIn = weights.kIn
    let kSteps = kIn / 4
    let nPanels = nOut / 8

    let bBase = weights.packedW.ptr
    let wsBase = weights.wScale.ptr
    let biasBase = weights.bias.ptr
    let kernels = AArch64AsmKernels.shared

    var accBuf = (
        SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero,
        SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero, SIMD4<Int32>.zero
    )

    withUnsafeMutablePointer(to: &accBuf) { accTuplePtr in
        let accI32 = UnsafeMutableRawPointer(accTuplePtr).assumingMemoryBound(to: Int32.self)
        let accVecs = UnsafeMutableRawPointer(accTuplePtr).assumingMemoryBound(to: int32x4_t.self)

        var r = 0
        while r + 4 <= m {
            let aRow0 = UnsafeRawPointer(aQ.advanced(by: r * kIn)).assumingMemoryBound(to: Int32.self)
            let aRow1 = UnsafeRawPointer(aQ.advanced(by: (r + 1) * kIn)).assumingMemoryBound(to: Int32.self)
            let aRow2 = UnsafeRawPointer(aQ.advanced(by: (r + 2) * kIn)).assumingMemoryBound(to: Int32.self)
            let aRow3 = UnsafeRawPointer(aQ.advanced(by: (r + 3) * kIn)).assumingMemoryBound(to: Int32.self)
            let s0 = vdupq_n_f32(aScale[r])
            let s1 = vdupq_n_f32(aScale[r + 1])
            let s2 = vdupq_n_f32(aScale[r + 2])
            let s3 = vdupq_n_f32(aScale[r + 3])
            let scales = (s0, s1, s2, s3)

            for nb in 0..<nPanels {
                let col = nb * 8
                let pb = bBase.advanced(by: nb * kSteps * 32)
                kernels.sdot4RowPanel(aRow0, aRow1, aRow2, aRow3, pb, kSteps, accI32)

                let wsLo = vld1q_f32(wsBase.advanced(by: col))
                let wsHi = vld1q_f32(wsBase.advanced(by: col + 4))
                let biasLo = vld1q_f32(biasBase.advanced(by: col))
                let biasHi = vld1q_f32(biasBase.advanced(by: col + 4))

                withUnsafePointer(to: scales) { scPtr in
                    let srVecs = UnsafeRawPointer(scPtr).assumingMemoryBound(to: float32x4_t.self)
                    for dr in 0..<4 {
                        let cLo = accVecs[2 * dr]
                        let cHi = accVecs[2 * dr + 1]
                        let sr = srVecs[dr]
                        let off = (r + dr) * nOut + col
                        var bLo = biasLo
                        var bHi = biasHi
                        if let rp = residual {
                            bLo = vaddq_f32(bLo, vld1q_f32(rp.advanced(by: off)))
                            bHi = vaddq_f32(bHi, vld1q_f32(rp.advanced(by: off + 4)))
                        }
                        vst1q_f32(out.advanced(by: off), vfmaq_f32(bLo, vcvtq_f32_s32(cLo), vmulq_f32(sr, wsLo)))
                        vst1q_f32(out.advanced(by: off + 4), vfmaq_f32(bHi, vcvtq_f32_s32(cHi), vmulq_f32(sr, wsHi)))
                    }
                }
            }
            r += 4
        }

        while r < m {
            let aRow = UnsafeRawPointer(aQ.advanced(by: r * kIn)).assumingMemoryBound(to: Int32.self)
            let sRow = vdupq_n_f32(aScale[r])
            let rowOff = r * nOut

            for nb in 0..<nPanels {
                let col = nb * 8
                let pb = bBase.advanced(by: nb * kSteps * 32)
                kernels.sdot1RowPanel(aRow, pb, kSteps, accI32)

                let c0 = accVecs[0]
                let c1 = accVecs[1]
                let wsLo = vmulq_f32(sRow, vld1q_f32(wsBase.advanced(by: col)))
                let wsHi = vmulq_f32(sRow, vld1q_f32(wsBase.advanced(by: col + 4)))
                var bLo = vld1q_f32(biasBase.advanced(by: col))
                var bHi = vld1q_f32(biasBase.advanced(by: col + 4))
                if let rp = residual {
                    bLo = vaddq_f32(bLo, vld1q_f32(rp.advanced(by: rowOff + col)))
                    bHi = vaddq_f32(bHi, vld1q_f32(rp.advanced(by: rowOff + col + 4)))
                }
                vst1q_f32(out.advanced(by: rowOff + col), vfmaq_f32(bLo, vcvtq_f32_s32(c0), wsLo))
                vst1q_f32(out.advanced(by: rowOff + col + 4), vfmaq_f32(bHi, vcvtq_f32_s32(c1), wsHi))
            }
            r += 1
        }
    }
}

@usableFromInline
func gemmNeonFused(
    m: Int,
    aQ: UnsafePointer<Int8>,
    aScale: UnsafePointer<Float>,
    weights: PackedLinearInt8,
    residual: UnsafePointer<Float>?,
    out: UnsafeMutablePointer<Float>
) {
    let nOut = weights.nOut
    let kIn = weights.kIn
    let wBase = weights.packedW.ptr
    let wsPtr = weights.wScale.ptr
    let biasPtr = weights.bias.ptr

    for r in 0..<m {
        let aRow = aQ.advanced(by: r * kIn)
        let sR = aScale[r]
        let rowOff = r * nOut
        for n in 0..<nOut {
            let wRow = wBase.advanced(by: n * kIn)
            var acc = vdupq_n_s32(0)
            var k = 0
            while k < kIn {
                let va = vld1q_s8(aRow.advanced(by: k))
                let vw = vld1q_s8(wRow.advanced(by: k))
                let pLo = vmull_s8(vget_low_s8(va), vget_low_s8(vw))
                let pHi = vmull_s8(vget_high_s8(va), vget_high_s8(vw))
                let sum16 = vaddq_s16(pLo, pHi)
                acc = vpadalq_s16(acc, sum16)
                k += 16
            }
            let dot = vaddvq_s32(acc)
            var bVal = biasPtr[n]
            if let res = residual {
                bVal += res[rowOff + n]
            }
            out[rowOff + n] = bVal.addingProduct(Float(dot), sR * wsPtr[n])
        }
    }
}
#endif

#if arch(x86_64)
@usableFromInline
func gemmX86Avx2Fused(
    m: Int,
    aQ: UnsafePointer<Int8>,
    aScale: UnsafePointer<Float>,
    weights: PackedLinearInt8,
    residual: UnsafePointer<Float>?,
    out: UnsafeMutablePointer<Float>
) {
    let nOut = weights.nOut
    let kIn = weights.kIn
    let kSteps = kIn / 4
    let nPanels = nOut / 16
    let bBase = weights.packedW.ptr
    let wsBase = weights.wScale.ptr
    let biasBase = weights.bias.ptr
    let panelFn = X86AsmKernels.shared.avx2Panel

    var acc16 = (
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
    )

    withUnsafeMutablePointer(to: &acc16) { accTuplePtr in
        let accPtr = UnsafeMutableRawPointer(accTuplePtr).assumingMemoryBound(to: Int32.self)
        for r in 0..<m {
            let aRow = UnsafeRawPointer(aQ.advanced(by: r * kIn)).assumingMemoryBound(to: Int32.self)
            let sR = aScale[r]
            let rowOff = r * nOut
            for nb in 0..<nPanels {
                let col = nb * 16
                let pb = bBase.advanced(by: nb * kSteps * 64)
                panelFn(aRow, pb, kSteps, accPtr)
                for j in 0..<16 {
                    let c = col + j
                    var bVal = biasBase[c]
                    if let res = residual {
                        bVal += res[rowOff + c]
                    }
                    out[rowOff + c] = bVal.addingProduct(Float(accPtr[j]), sR * wsBase[c])
                }
            }
        }
    }
}

@usableFromInline
func gemmX86AvxVnniFused(
    m: Int,
    aQ: UnsafePointer<Int8>,
    aU8Scratch: UnsafeMutablePointer<Int8>,
    aScale: UnsafePointer<Float>,
    weights: PackedLinearInt8,
    residual: UnsafePointer<Float>?,
    out: UnsafeMutablePointer<Float>
) {
    let nOut = weights.nOut
    let kIn = weights.kIn
    let kSteps = kIn / 4
    let nPanels = nOut / 16
    let totalA = m * kIn
    for idx in 0..<totalA {
        aU8Scratch[idx] = Int8(bitPattern: UInt8(bitPattern: aQ[idx]) ^ 0x80)
    }

    let bBase = weights.packedW.ptr
    let compBase = weights.colSumComp.ptr
    let wsBase = weights.wScale.ptr
    let biasBase = weights.bias.ptr
    let panelFn = X86AsmKernels.shared.avxVnniPanel

    var acc16 = (
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
    )

    withUnsafeMutablePointer(to: &acc16) { accTuplePtr in
        let accPtr = UnsafeMutableRawPointer(accTuplePtr).assumingMemoryBound(to: Int32.self)
        for r in 0..<m {
            let aRow = UnsafeRawPointer(aU8Scratch.advanced(by: r * kIn)).assumingMemoryBound(to: Int32.self)
            let sR = aScale[r]
            let rowOff = r * nOut
            for nb in 0..<nPanels {
                let col = nb * 16
                let pb = bBase.advanced(by: nb * kSteps * 64)
                let compCol = compBase.advanced(by: col)
                panelFn(aRow, pb, compCol, kSteps, accPtr)
                for j in 0..<16 {
                    let c = col + j
                    var bVal = biasBase[c]
                    if let res = residual {
                        bVal += res[rowOff + c]
                    }
                    out[rowOff + c] = bVal.addingProduct(Float(accPtr[j]), sR * wsBase[c])
                }
            }
        }
    }
}
#endif

public func gemmScalarFused(
    m: Int,
    aQ: UnsafePointer<Int8>,
    aScale: UnsafePointer<Float>,
    weights: PackedLinearInt8,
    residual: UnsafePointer<Float>?,
    out: UnsafeMutablePointer<Float>
) {
    let nOut = weights.nOut
    let kIn = weights.kIn
    let w = weights.packedW.ptr
    let wsPtr = weights.wScale.ptr
    let biasPtr = weights.bias.ptr

    for r in 0..<m {
        let aRow = aQ.advanced(by: r * kIn)
        let sR = aScale[r]
        let rowOff = r * nOut
        for n in 0..<nOut {
            let wRow = w.advanced(by: n * kIn)
            var acc: Int32 = 0
            for k in 0..<kIn {
                acc &+= Int32(aRow[k]) &* Int32(wRow[k])
            }
            var bVal = biasPtr[n]
            if let res = residual {
                bVal += res[rowOff + n]
            }
            out[rowOff + n] = bVal.addingProduct(Float(acc), sR * wsPtr[n])
        }
    }
}
