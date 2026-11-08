import Foundation
#if arch(arm64)
import _Builtin_intrinsics.arm.neon
#endif

@usableFromInline let LOG2_E_F32: Float = 1.44269504088896340735992468100189214
@usableFromInline let FRAC_1_SQRT_2_F32: Float = 0.707106781186547524400844362104849039

@inlinable
@inline(__always)
public func scalarGelu(_ x: Float) -> Float {
    let z = abs(x) * FRAC_1_SQRT_2_F32
    let t: Float = 1.0 / (1.0 + 0.3275911 * z)
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    let erfMag = 1.0 - poly * Foundation.exp(-z * z)
    let erfVal = Float(signOf: x, magnitudeOf: erfMag)
    return 0.5 * x * (1.0 + erfVal)
}

#if arch(arm64)
@usableFromInline
@inline(__always)
func neonExpF32(_ yIn: float32x4_t) -> float32x4_t {
    let y = vmaxq_f32(vminq_f32(yIn, vdupq_n_f32(88.0)), vdupq_n_f32(-87.3))
    let log2e = vdupq_n_f32(LOG2_E_F32)
    let ln2Hi = vdupq_n_f32(0.693145751953125)
    let ln2Lo = vdupq_n_f32(1.428606765330187e-6)

    let n = vrndnq_f32(vmulq_f32(y, log2e))
    let r = vfmsq_f32(vfmsq_f32(y, n, ln2Hi), n, ln2Lo)

    let c5 = vdupq_n_f32(0.008333334)
    let c4 = vdupq_n_f32(0.041666668)
    let c3 = vdupq_n_f32(0.16666667)
    let c2 = vdupq_n_f32(0.5)
    let c1 = vdupq_n_f32(1.0)

    var p = vfmaq_f32(c4, r, c5)
    p = vfmaq_f32(c3, r, p)
    p = vfmaq_f32(c2, r, p)
    p = vfmaq_f32(c1, r, p)
    p = vfmaq_f32(c1, r, p)

    let ni = vaddq_s32(vcvtnq_s32_f32(n), vdupq_n_s32(127))
    let pow2n = vreinterpretq_f32_s32(ni &<< 23)
    return vmulq_f32(p, pow2n)
}

@usableFromInline
@inline(__always)
func neonGeluPairF32(_ x0: float32x4_t, _ x1: float32x4_t) -> (float32x4_t, float32x4_t) {
    let vmin = vdupq_n_f32(-4.8)
    let vmax = vdupq_n_f32(4.8)
    let xc0 = vminq_f32(vmaxq_f32(x0, vmin), vmax)
    let xc1 = vminq_f32(vmaxq_f32(x1, vmin), vmax)
    let u0 = vmulq_f32(xc0, xc0)
    let u1 = vmulq_f32(xc1, xc1)

    let p3 = vdupq_n_f32(3.214524e-5)
    let p2 = vdupq_n_f32(0.0037863153)
    let p1 = vdupq_n_f32(0.027641433)
    let p0 = vdupq_n_f32(0.39894123)

    var num0 = vfmaq_f32(p2, u0, p3)
    var num1 = vfmaq_f32(p2, u1, p3)
    num0 = vfmaq_f32(p1, u0, num0)
    num1 = vfmaq_f32(p1, u1, num1)
    num0 = vfmaq_f32(p0, u0, num0)
    num1 = vfmaq_f32(p0, u1, num1)

    let q3 = vdupq_n_f32(0.001139532)
    let q2 = vdupq_n_f32(0.02381135)
    let q1 = vdupq_n_f32(0.2359431)
    let one = vdupq_n_f32(1.0)

    var den0 = vfmaq_f32(q2, u0, q3)
    var den1 = vfmaq_f32(q2, u1, q3)
    den0 = vfmaq_f32(q1, u0, den0)
    den1 = vfmaq_f32(q1, u1, den1)
    den0 = vfmaq_f32(one, u0, den0)
    den1 = vfmaq_f32(one, u1, den1)

    let est0 = vrecpeq_f32(den0)
    let est1 = vrecpeq_f32(den1)
    let inv0 = vmulq_f32(est0, vrecpsq_f32(den0, est0))
    let inv1 = vmulq_f32(est1, vrecpsq_f32(den1, est1))

    let half = vdupq_n_f32(0.5)
    let gate0 = vfmaq_f32(half, xc0, vmulq_f32(num0, inv0))
    let gate1 = vfmaq_f32(half, xc1, vmulq_f32(num1, inv1))
    return (vmulq_f32(x0, gate0), vmulq_f32(x1, gate1))
}
#endif

// MARK: - LayerNorm + SmoothQuant + Dynamic Int8 Quantization

@inlinable
public func layernormSmoothQuantizeRow(
    xRow: UnsafeMutablePointer<Float>,
    count n: Int,
    gamma: UnsafePointer<Float>,
    beta: UnsafePointer<Float>,
    invSmooth: UnsafePointer<Float>,
    outQ: UnsafeMutablePointer<Int8>,
    eps: Float
) -> Float {
    #if arch(arm64)
    return layernormSmoothQuantizeRowNeon(
        xRow: xRow,
        count: n,
        gamma: gamma,
        beta: beta,
        invSmooth: invSmooth,
        outQ: outQ,
        eps: eps
    )
    #else
    return layernormSmoothQuantizeRowScalar(
        xRow: xRow,
        count: n,
        gamma: gamma,
        beta: beta,
        invSmooth: invSmooth,
        outQ: outQ,
        eps: eps
    )
    #endif
}

@inlinable
public func layernormSmoothQuantizeRowScalar(
    xRow: UnsafeMutablePointer<Float>,
    count n: Int,
    gamma: UnsafePointer<Float>,
    beta: UnsafePointer<Float>,
    invSmooth: UnsafePointer<Float>,
    outQ: UnsafeMutablePointer<Int8>,
    eps: Float
) -> Float {
    var sum: Float = 0.0
    for i in 0..<n {
        sum += xRow[i]
    }
    let mean = sum / Float(n)
    var varSum: Float = 0.0
    for i in 0..<n {
        let d = xRow[i] - mean
        varSum += d * d
    }
    let rstd = 1.0 / (varSum / Float(n) + eps).squareRoot()

    var maxAbs: Float = 1e-8
    for j in 0..<n {
        let y = (xRow[j] - mean) * rstd * gamma[j] + beta[j]
        xRow[j] = y
        let a = abs(y * invSmooth[j])
        if a > maxAbs {
            maxAbs = a
        }
    }
    let scale = maxAbs / 127.0
    let invScale = 127.0 / maxAbs
    for j in 0..<n {
        let v = (xRow[j] * invSmooth[j] * invScale).rounded(.toNearestOrAwayFromZero)
        outQ[j] = Int8(min(max(v, -127.0), 127.0))
    }
    return scale
}

#if arch(arm64)
@usableFromInline
func layernormSmoothQuantizeRowNeon(
    xRow: UnsafeMutablePointer<Float>,
    count n: Int,
    gamma: UnsafePointer<Float>,
    beta: UnsafePointer<Float>,
    invSmooth: UnsafePointer<Float>,
    outQ: UnsafeMutablePointer<Int8>,
    eps: Float
) -> Float {
    var sum0 = vdupq_n_f32(0.0)
    var sum1 = vdupq_n_f32(0.0)
    var sum2 = vdupq_n_f32(0.0)
    var sum3 = vdupq_n_f32(0.0)
    var i = 0
    while i < n {
        sum0 = vaddq_f32(sum0, vld1q_f32(xRow.advanced(by: i)))
        sum1 = vaddq_f32(sum1, vld1q_f32(xRow.advanced(by: i + 4)))
        sum2 = vaddq_f32(sum2, vld1q_f32(xRow.advanced(by: i + 8)))
        sum3 = vaddq_f32(sum3, vld1q_f32(xRow.advanced(by: i + 12)))
        i += 16
    }
    let mean = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3))) / Float(n)
    let vmean = vdupq_n_f32(mean)

    var vsum0 = vdupq_n_f32(0.0)
    var vsum1 = vdupq_n_f32(0.0)
    var vsum2 = vdupq_n_f32(0.0)
    var vsum3 = vdupq_n_f32(0.0)
    i = 0
    while i < n {
        let d0 = vsubq_f32(vld1q_f32(xRow.advanced(by: i)), vmean)
        let d1 = vsubq_f32(vld1q_f32(xRow.advanced(by: i + 4)), vmean)
        let d2 = vsubq_f32(vld1q_f32(xRow.advanced(by: i + 8)), vmean)
        let d3 = vsubq_f32(vld1q_f32(xRow.advanced(by: i + 12)), vmean)
        vsum0 = vfmaq_f32(vsum0, d0, d0)
        vsum1 = vfmaq_f32(vsum1, d1, d1)
        vsum2 = vfmaq_f32(vsum2, d2, d2)
        vsum3 = vfmaq_f32(vsum3, d3, d3)
        i += 16
    }
    let variance = vaddvq_f32(vaddq_f32(vaddq_f32(vsum0, vsum1), vaddq_f32(vsum2, vsum3))) / Float(n)
    let vrstd = vdupq_n_f32(1.0 / (variance + eps).squareRoot())

    var vmax0 = vdupq_n_f32(1e-8)
    var vmax1 = vdupq_n_f32(1e-8)
    var j = 0
    while j < n {
        let y0 = vfmaq_f32(
            vld1q_f32(beta.advanced(by: j)),
            vmulq_f32(vsubq_f32(vld1q_f32(xRow.advanced(by: j)), vmean), vrstd),
            vld1q_f32(gamma.advanced(by: j))
        )
        let y1 = vfmaq_f32(
            vld1q_f32(beta.advanced(by: j + 4)),
            vmulq_f32(vsubq_f32(vld1q_f32(xRow.advanced(by: j + 4)), vmean), vrstd),
            vld1q_f32(gamma.advanced(by: j + 4))
        )
        let y2 = vfmaq_f32(
            vld1q_f32(beta.advanced(by: j + 8)),
            vmulq_f32(vsubq_f32(vld1q_f32(xRow.advanced(by: j + 8)), vmean), vrstd),
            vld1q_f32(gamma.advanced(by: j + 8))
        )
        let y3 = vfmaq_f32(
            vld1q_f32(beta.advanced(by: j + 12)),
            vmulq_f32(vsubq_f32(vld1q_f32(xRow.advanced(by: j + 12)), vmean), vrstd),
            vld1q_f32(gamma.advanced(by: j + 12))
        )
        vst1q_f32(xRow.advanced(by: j), y0)
        vst1q_f32(xRow.advanced(by: j + 4), y1)
        vst1q_f32(xRow.advanced(by: j + 8), y2)
        vst1q_f32(xRow.advanced(by: j + 12), y3)

        let ys0 = vmulq_f32(y0, vld1q_f32(invSmooth.advanced(by: j)))
        let ys1 = vmulq_f32(y1, vld1q_f32(invSmooth.advanced(by: j + 4)))
        let ys2 = vmulq_f32(y2, vld1q_f32(invSmooth.advanced(by: j + 8)))
        let ys3 = vmulq_f32(y3, vld1q_f32(invSmooth.advanced(by: j + 12)))
        vmax0 = vmaxq_f32(vmax0, vmaxq_f32(vabsq_f32(ys0), vabsq_f32(ys1)))
        vmax1 = vmaxq_f32(vmax1, vmaxq_f32(vabsq_f32(ys2), vabsq_f32(ys3)))
        j += 16
    }

    let maxAbs = vmaxvq_f32(vmaxq_f32(vmax0, vmax1))
    let scale = maxAbs / 127.0
    let vinvScale = vdupq_n_f32(127.0 / maxAbs)

    j = 0
    while j < n {
        let ys0 = vmulq_f32(vld1q_f32(xRow.advanced(by: j)), vld1q_f32(invSmooth.advanced(by: j)))
        let ys1 = vmulq_f32(vld1q_f32(xRow.advanced(by: j + 4)), vld1q_f32(invSmooth.advanced(by: j + 4)))
        let ys2 = vmulq_f32(vld1q_f32(xRow.advanced(by: j + 8)), vld1q_f32(invSmooth.advanced(by: j + 8)))
        let ys3 = vmulq_f32(vld1q_f32(xRow.advanced(by: j + 12)), vld1q_f32(invSmooth.advanced(by: j + 12)))
        let q0 = vcvtnq_s32_f32(vmulq_f32(ys0, vinvScale))
        let q1 = vcvtnq_s32_f32(vmulq_f32(ys1, vinvScale))
        let q2 = vcvtnq_s32_f32(vmulq_f32(ys2, vinvScale))
        let q3 = vcvtnq_s32_f32(vmulq_f32(ys3, vinvScale))

        let q01 = vqmovn_high_s32(vqmovn_s32(q0), q1)
        let q23 = vqmovn_high_s32(vqmovn_s32(q2), q3)
        let qBytes = vqmovn_high_s16(vqmovn_s16(q01), q23)
        vst1q_s8(outQ.advanced(by: j), qBytes)
        j += 16
    }
    return scale
}
#endif

/// In-place LayerNorm without quantization (used at the end of Layer 11).
@inlinable
public func layernormInplaceRow(
    xRow: UnsafeMutablePointer<Float>,
    count n: Int,
    gamma: UnsafePointer<Float>,
    beta: UnsafePointer<Float>,
    eps: Float
) {
    var sum: Float = 0.0
    for i in 0..<n {
        sum += xRow[i]
    }
    let mean = sum / Float(n)
    var varSum: Float = 0.0
    for i in 0..<n {
        let d = xRow[i] - mean
        varSum += d * d
    }
    let rstd = 1.0 / (varSum / Float(n) + eps).squareRoot()
    for j in 0..<n {
        xRow[j] = (xRow[j] - mean) * rstd * gamma[j] + beta[j]
    }
}

/// SmoothQuant scaling + Dynamic Int8 quantization for a single row (`[384]`).
@inlinable
public func smoothQuantizeRow(
    xRow: UnsafePointer<Float>,
    count n: Int,
    invSmooth: UnsafePointer<Float>,
    outQ: UnsafeMutablePointer<Int8>
) -> Float {
    #if arch(arm64)
    return smoothQuantizeRowNeon(xRow: xRow, count: n, invSmooth: invSmooth, outQ: outQ)
    #else
    var maxAbs: Float = 1e-8
    for j in 0..<n {
        let a = abs(xRow[j] * invSmooth[j])
        if a > maxAbs {
            maxAbs = a
        }
    }
    let scale = maxAbs / 127.0
    let invScale = 127.0 / maxAbs
    for j in 0..<n {
        let v = (xRow[j] * invSmooth[j] * invScale).rounded(.toNearestOrAwayFromZero)
        outQ[j] = Int8(min(max(v, -127.0), 127.0))
    }
    return scale
    #endif
}

#if arch(arm64)
@usableFromInline
@inline(__always)
func smoothQuantizeRowNeon(
    xRow: UnsafePointer<Float>,
    count n: Int,
    invSmooth: UnsafePointer<Float>,
    outQ: UnsafeMutablePointer<Int8>
) -> Float {
    var vmax0 = vdupq_n_f32(1e-8)
    var vmax1 = vdupq_n_f32(1e-8)
    var i = 0
    while i < n {
        let ys0 = vmulq_f32(vld1q_f32(xRow.advanced(by: i)), vld1q_f32(invSmooth.advanced(by: i)))
        let ys1 = vmulq_f32(vld1q_f32(xRow.advanced(by: i + 4)), vld1q_f32(invSmooth.advanced(by: i + 4)))
        let ys2 = vmulq_f32(vld1q_f32(xRow.advanced(by: i + 8)), vld1q_f32(invSmooth.advanced(by: i + 8)))
        let ys3 = vmulq_f32(vld1q_f32(xRow.advanced(by: i + 12)), vld1q_f32(invSmooth.advanced(by: i + 12)))
        vmax0 = vmaxq_f32(vmax0, vmaxq_f32(vabsq_f32(ys0), vabsq_f32(ys1)))
        vmax1 = vmaxq_f32(vmax1, vmaxq_f32(vabsq_f32(ys2), vabsq_f32(ys3)))
        i += 16
    }

    let maxAbs = vmaxvq_f32(vmaxq_f32(vmax0, vmax1))
    let scale = maxAbs / 127.0
    let vinvScale = vdupq_n_f32(127.0 / maxAbs)

    i = 0
    while i < n {
        let ys0 = vmulq_f32(vld1q_f32(xRow.advanced(by: i)), vld1q_f32(invSmooth.advanced(by: i)))
        let ys1 = vmulq_f32(vld1q_f32(xRow.advanced(by: i + 4)), vld1q_f32(invSmooth.advanced(by: i + 4)))
        let ys2 = vmulq_f32(vld1q_f32(xRow.advanced(by: i + 8)), vld1q_f32(invSmooth.advanced(by: i + 8)))
        let ys3 = vmulq_f32(vld1q_f32(xRow.advanced(by: i + 12)), vld1q_f32(invSmooth.advanced(by: i + 12)))
        let q0 = vcvtnq_s32_f32(vmulq_f32(ys0, vinvScale))
        let q1 = vcvtnq_s32_f32(vmulq_f32(ys1, vinvScale))
        let q2 = vcvtnq_s32_f32(vmulq_f32(ys2, vinvScale))
        let q3 = vcvtnq_s32_f32(vmulq_f32(ys3, vinvScale))

        let q01 = vqmovn_high_s32(vqmovn_s32(q0), q1)
        let q23 = vqmovn_high_s32(vqmovn_s32(q2), q3)
        vst1q_s8(outQ.advanced(by: i), vqmovn_high_s16(vqmovn_s16(q01), q23))
        i += 16
    }
    return scale
}
#endif

// MARK: - GELU + SmoothQuant + Dynamic Int8 Quantization

#if arch(arm64)
private let GELU_F16_MAX_N: Int = 1536
private let GELU_F16_CONSTS_BUF: AlignedBuf<Float> = AlignedBuf<Float>.fromSlice([
    -4.8,
    4.8,
    3.214524e-5,
    0.0037863153,
    0.027641433,
    0.39894123,
    0.001139532,
    0.02381135,
    0.2359431,
    1.0,
    0.5,
    0.0,
])

private let HAS_FP16_ARITH: Bool = {
    detectArmFeature(darwinSysctl: "hw.optional.arm.FEAT_FP16", linuxHwcap1Bit: 9)
}()

@usableFromInline
@inline(__always)
func hasFp16Arith() -> Bool {
    HAS_FP16_ARITH
}

@usableFromInline
@inline(__always)
func geluF16Pass1Wrapper(
    x: UnsafeMutablePointer<Float>,
    s16: UnsafePointer<UInt16>,
    n: Int
) -> Float {
    let bits = AArch64AsmKernels.shared.geluF16Pass1(GELU_F16_CONSTS_BUF.ptr, x, s16, n / 16)
    return Float(bitPattern: bits)
}

@usableFromInline
@discardableResult
func geluF16Pass2Wrapper(
    x: UnsafeMutablePointer<Float>,
    n: Int,
    maxAbs: Float,
    q: UnsafeMutablePointer<Int8>
) -> Float {
    let h = UnsafeRawPointer(x).assumingMemoryBound(to: UInt16.self)
    if !(maxAbs >= 1e-2 && maxAbs.isFinite) {
        let mx = maxAbs.isFinite ? max(maxAbs, 1e-8) : Float.greatestFiniteMagnitude
        let inv: Float = 127.0 / mx
        var tmp = (Float(0), Float(0), Float(0), Float(0), Float(0), Float(0), Float(0), Float(0))
        withUnsafeMutablePointer(to: &tmp) { tmpTuplePtr in
            let tmpPtr = UnsafeMutableRawPointer(tmpTuplePtr).assumingMemoryBound(to: Float.self)
            var j = 0
            while j < n {
                AArch64AsmKernels.shared.f16x8ToF32x8(h.advanced(by: j), tmpPtr)
                for k in 0..<8 {
                    let v = tmpPtr[k] * inv
                    if v.isNaN {
                        q[j + k] = 0
                    } else {
                        let r = v.rounded(.toNearestOrAwayFromZero)
                        q[j + k] = Int8(min(max(r, -127.0), 127.0))
                    }
                }
                j += 8
            }
        }
        return mx / 127.0
    }
    let invScale: Float = 127.0 / maxAbs
    AArch64AsmKernels.shared.geluF16Pass2(invScale.bitPattern, h, q, n / 16)
    return maxAbs / 127.0
}

@usableFromInline
func geluSmoothQuantizeBatchF16(
    m: Int,
    interBuf: UnsafeMutablePointer<Float>,
    invSmooth: UnsafePointer<Float>,
    n: Int,
    interQ: UnsafeMutablePointer<Int8>,
    actScale: UnsafeMutablePointer<Float>
) {
    withUnsafeTemporaryAllocation(of: UInt16.self, capacity: n) { s16Buf in
        let s16Ptr = s16Buf.baseAddress!
        let kernels = AArch64AsmKernels.shared
        let cPtr = GELU_F16_CONSTS_BUF.ptr
        kernels.f32ToF16Neon(invSmooth, s16Ptr, n / 16)
        for r in 0..<m {
            let xRow = interBuf.advanced(by: r * n)
            let maxAbs = Float(bitPattern: kernels.geluF16Pass1(cPtr, xRow, s16Ptr, n / 16))
            actScale[r] = geluF16Pass2Wrapper(x: xRow, n: n, maxAbs: maxAbs, q: interQ.advanced(by: r * n))
        }
    }
}

public func geluSegmentPass1(
    rows: Int,
    x: UnsafeMutablePointer<Float>,
    rowStride: Int,
    col0: Int,
    nCols: Int,
    invSmooth: UnsafePointer<Float>,
    rowMax: UnsafeMutablePointer<Float>
) {
    if hasFp16Arith() {
        withUnsafeTemporaryAllocation(of: UInt16.self, capacity: nCols) { s16Buf in
            let s16Ptr = s16Buf.baseAddress!
            let kernels = AArch64AsmKernels.shared
            let cPtr = GELU_F16_CONSTS_BUF.ptr
            kernels.f32ToF16Neon(invSmooth.advanced(by: col0), s16Ptr, nCols / 16)
            for r in 0..<rows {
                let bits = kernels.geluF16Pass1(cPtr, x.advanced(by: r * rowStride + col0), s16Ptr, nCols / 16)
                rowMax[r] = Float(bitPattern: bits)
            }
        }
    } else {
        for r in 0..<rows {
            let seg = x.advanced(by: r * rowStride + col0)
            var mx: Float = 0.0
            var j = 0
            while j < nCols {
                let (g0Raw, g1Raw) = neonGeluPairF32(
                    vld1q_f32(seg.advanced(by: j)),
                    vld1q_f32(seg.advanced(by: j + 4))
                )
                let g0 = vmulq_f32(g0Raw, vld1q_f32(invSmooth.advanced(by: col0 + j)))
                let g1 = vmulq_f32(g1Raw, vld1q_f32(invSmooth.advanced(by: col0 + j + 4)))
                vst1q_f32(seg.advanced(by: j), g0)
                vst1q_f32(seg.advanced(by: j + 4), g1)
                let blockMax = vmaxvq_f32(vmaxq_f32(vabsq_f32(g0), vabsq_f32(g1)))
                if blockMax > mx { mx = blockMax }
                j += 8
            }
            rowMax[r] = mx
        }
    }
}

public func geluSegmentQuantize(
    rows: Int,
    x: UnsafeMutablePointer<Float>,
    rowStride: Int,
    col0: Int,
    nCols: Int,
    rowMaxFull: UnsafePointer<Float>,
    q: UnsafeMutablePointer<Int8>
) {
    let f16 = hasFp16Arith()
    for r in 0..<rows {
        let xs = x.advanced(by: r * rowStride + col0)
        let qs = q.advanced(by: r * rowStride + col0)
        if f16 {
            geluF16Pass2Wrapper(x: xs, n: nCols, maxAbs: rowMaxFull[r], q: qs)
        } else {
            let mx = max(rowMaxFull[r], 1e-8)
            let inv = vdupq_n_f32(127.0 / mx)
            var j = 0
            while j < nCols {
                let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xs.advanced(by: j)), inv))
                let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xs.advanced(by: j + 4)), inv))
                let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xs.advanced(by: j + 8)), inv))
                let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xs.advanced(by: j + 12)), inv))
                let q01 = vqmovn_high_s32(vqmovn_s32(q0), q1)
                let q23 = vqmovn_high_s32(vqmovn_s32(q2), q3)
                vst1q_s8(qs.advanced(by: j), vqmovn_high_s16(vqmovn_s16(q01), q23))
                j += 16
            }
        }
    }
}
#endif

@inlinable
public func geluSmoothQuantizeBatch(
    m: Int,
    interBuf: UnsafeMutablePointer<Float>,
    invSmooth: UnsafePointer<Float>,
    nCols n: Int,
    interQ: UnsafeMutablePointer<Int8>,
    actScale: UnsafeMutablePointer<Float>,
    useSme: Bool
) {
    #if arch(arm64)
    if n % 16 == 0 && n <= 1536 && hasFp16Arith() {
        geluSmoothQuantizeBatchF16(
            m: m,
            interBuf: interBuf,
            invSmooth: invSmooth,
            n: n,
            interQ: interQ,
            actScale: actScale
        )
        return
    }
    #endif
    for r in 0..<m {
        let row = interBuf.advanced(by: r * n)
        let qRow = interQ.advanced(by: r * n)
        actScale[r] = geluSmoothQuantizeRow(xRow: row, count: n, invSmooth: invSmooth, outQ: qRow)
    }
}

@inlinable
public func geluSmoothQuantizeRow(
    xRow: UnsafeMutablePointer<Float>,
    count n: Int,
    invSmooth: UnsafePointer<Float>,
    outQ: UnsafeMutablePointer<Int8>
) -> Float {
    #if arch(arm64)
    return geluSmoothQuantizeRowNeon(xRow: xRow, count: n, invSmooth: invSmooth, outQ: outQ)
    #else
    var maxAbs: Float = 1e-8
    for j in 0..<n {
        let g = scalarGelu(xRow[j]) * invSmooth[j]
        xRow[j] = g
        let a = abs(g)
        if a > maxAbs {
            maxAbs = a
        }
    }
    let scale = maxAbs / 127.0
    let invScale = 127.0 / maxAbs
    for j in 0..<n {
        let v = (xRow[j] * invScale).rounded(.toNearestOrAwayFromZero)
        outQ[j] = Int8(min(max(v, -127.0), 127.0))
    }
    return scale
    #endif
}

#if arch(arm64)
@usableFromInline
@inline(__always)
func geluSmoothQuantizeRowNeon(
    xRow: UnsafeMutablePointer<Float>,
    count n: Int,
    invSmooth: UnsafePointer<Float>,
    outQ: UnsafeMutablePointer<Int8>
) -> Float {
    var vmax0 = vdupq_n_f32(1e-8)
    var vmax1 = vdupq_n_f32(1e-8)
    var i = 0
    while i < n {
        let (g0Raw, g1Raw) = neonGeluPairF32(
            vld1q_f32(xRow.advanced(by: i)),
            vld1q_f32(xRow.advanced(by: i + 4))
        )
        let g0 = vmulq_f32(g0Raw, vld1q_f32(invSmooth.advanced(by: i)))
        let g1 = vmulq_f32(g1Raw, vld1q_f32(invSmooth.advanced(by: i + 4)))
        vst1q_f32(xRow.advanced(by: i), g0)
        vst1q_f32(xRow.advanced(by: i + 4), g1)
        vmax0 = vmaxq_f32(vmax0, vmaxq_f32(vabsq_f32(g0), vabsq_f32(g1)))

        let (g2Raw, g3Raw) = neonGeluPairF32(
            vld1q_f32(xRow.advanced(by: i + 8)),
            vld1q_f32(xRow.advanced(by: i + 12))
        )
        let g2 = vmulq_f32(g2Raw, vld1q_f32(invSmooth.advanced(by: i + 8)))
        let g3 = vmulq_f32(g3Raw, vld1q_f32(invSmooth.advanced(by: i + 12)))
        vst1q_f32(xRow.advanced(by: i + 8), g2)
        vst1q_f32(xRow.advanced(by: i + 12), g3)
        vmax1 = vmaxq_f32(vmax1, vmaxq_f32(vabsq_f32(g2), vabsq_f32(g3)))
        i += 16
    }

    let maxAbs = vmaxvq_f32(vmaxq_f32(vmax0, vmax1))
    let scale = maxAbs / 127.0
    let vinvScale = vdupq_n_f32(127.0 / maxAbs)

    i = 0
    while i < n {
        let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xRow.advanced(by: i)), vinvScale))
        let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xRow.advanced(by: i + 4)), vinvScale))
        let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xRow.advanced(by: i + 8)), vinvScale))
        let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xRow.advanced(by: i + 12)), vinvScale))

        let q01 = vqmovn_high_s32(vqmovn_s32(q0), q1)
        let q23 = vqmovn_high_s32(vqmovn_s32(q2), q3)
        vst1q_s8(outQ.advanced(by: i), vqmovn_high_s16(vqmovn_s16(q01), q23))
        i += 16
    }
    return scale
}
#endif

// MARK: - Attention Math Kernels (head_dim = 32)

@inlinable
@inline(__always)
public func dot32F32(a: UnsafePointer<Float>, b: UnsafePointer<Float>) -> Float {
    #if arch(arm64)
    var s0 = vmulq_f32(vld1q_f32(a), vld1q_f32(b))
    var s1 = vmulq_f32(vld1q_f32(a.advanced(by: 4)), vld1q_f32(b.advanced(by: 4)))
    var s2 = vmulq_f32(vld1q_f32(a.advanced(by: 8)), vld1q_f32(b.advanced(by: 8)))
    var s3 = vmulq_f32(vld1q_f32(a.advanced(by: 12)), vld1q_f32(b.advanced(by: 12)))
    s0 = vfmaq_f32(s0, vld1q_f32(a.advanced(by: 16)), vld1q_f32(b.advanced(by: 16)))
    s1 = vfmaq_f32(s1, vld1q_f32(a.advanced(by: 20)), vld1q_f32(b.advanced(by: 20)))
    s2 = vfmaq_f32(s2, vld1q_f32(a.advanced(by: 24)), vld1q_f32(b.advanced(by: 24)))
    s3 = vfmaq_f32(s3, vld1q_f32(a.advanced(by: 28)), vld1q_f32(b.advanced(by: 28)))
    return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)))
    #else
    var sum: Float = 0.0
    for i in 0..<32 {
        sum += a[i] * b[i]
    }
    return sum
    #endif
}

@inlinable
public func computeScoresHead32(
    qSlice: UnsafePointer<Float>,
    kHead: UnsafePointer<Float>,
    scale: Float,
    scores: UnsafeMutablePointer<Float>,
    seqLen: Int
) {
    #if arch(arm64)
    let vscale = vdupq_n_f32(scale)
    let q0 = vmulq_f32(vld1q_f32(qSlice), vscale)
    let q1 = vmulq_f32(vld1q_f32(qSlice.advanced(by: 4)), vscale)
    let q2 = vmulq_f32(vld1q_f32(qSlice.advanced(by: 8)), vscale)
    let q3 = vmulq_f32(vld1q_f32(qSlice.advanced(by: 12)), vscale)
    let q4 = vmulq_f32(vld1q_f32(qSlice.advanced(by: 16)), vscale)
    let q5 = vmulq_f32(vld1q_f32(qSlice.advanced(by: 20)), vscale)
    let q6 = vmulq_f32(vld1q_f32(qSlice.advanced(by: 24)), vscale)
    let q7 = vmulq_f32(vld1q_f32(qSlice.advanced(by: 28)), vscale)

    var tk = 0
    while tk + 4 <= seqLen {
        let k0 = kHead.advanced(by: tk * 32)
        let k1 = k0.advanced(by: 32)
        let k2 = k0.advanced(by: 64)
        let k3 = k0.advanced(by: 96)

        var s0 = vaddq_f32(vmulq_f32(q0, vld1q_f32(k0)), vmulq_f32(q1, vld1q_f32(k0.advanced(by: 4))))
        var s1 = vaddq_f32(vmulq_f32(q0, vld1q_f32(k1)), vmulq_f32(q1, vld1q_f32(k1.advanced(by: 4))))
        var s2 = vaddq_f32(vmulq_f32(q0, vld1q_f32(k2)), vmulq_f32(q1, vld1q_f32(k2.advanced(by: 4))))
        var s3 = vaddq_f32(vmulq_f32(q0, vld1q_f32(k3)), vmulq_f32(q1, vld1q_f32(k3.advanced(by: 4))))

        s0 = vfmaq_f32(s0, q2, vld1q_f32(k0.advanced(by: 8)))
        s1 = vfmaq_f32(s1, q2, vld1q_f32(k1.advanced(by: 8)))
        s2 = vfmaq_f32(s2, q2, vld1q_f32(k2.advanced(by: 8)))
        s3 = vfmaq_f32(s3, q2, vld1q_f32(k3.advanced(by: 8)))

        s0 = vfmaq_f32(s0, q3, vld1q_f32(k0.advanced(by: 12)))
        s1 = vfmaq_f32(s1, q3, vld1q_f32(k1.advanced(by: 12)))
        s2 = vfmaq_f32(s2, q3, vld1q_f32(k2.advanced(by: 12)))
        s3 = vfmaq_f32(s3, q3, vld1q_f32(k3.advanced(by: 12)))

        s0 = vfmaq_f32(s0, q4, vld1q_f32(k0.advanced(by: 16)))
        s1 = vfmaq_f32(s1, q4, vld1q_f32(k1.advanced(by: 16)))
        s2 = vfmaq_f32(s2, q4, vld1q_f32(k2.advanced(by: 16)))
        s3 = vfmaq_f32(s3, q4, vld1q_f32(k3.advanced(by: 16)))

        s0 = vfmaq_f32(s0, q5, vld1q_f32(k0.advanced(by: 20)))
        s1 = vfmaq_f32(s1, q5, vld1q_f32(k1.advanced(by: 20)))
        s2 = vfmaq_f32(s2, q5, vld1q_f32(k2.advanced(by: 20)))
        s3 = vfmaq_f32(s3, q5, vld1q_f32(k3.advanced(by: 20)))

        s0 = vfmaq_f32(s0, q6, vld1q_f32(k0.advanced(by: 24)))
        s1 = vfmaq_f32(s1, q6, vld1q_f32(k1.advanced(by: 24)))
        s2 = vfmaq_f32(s2, q6, vld1q_f32(k2.advanced(by: 24)))
        s3 = vfmaq_f32(s3, q6, vld1q_f32(k3.advanced(by: 24)))

        s0 = vfmaq_f32(s0, q7, vld1q_f32(k0.advanced(by: 28)))
        s1 = vfmaq_f32(s1, q7, vld1q_f32(k1.advanced(by: 28)))
        s2 = vfmaq_f32(s2, q7, vld1q_f32(k2.advanced(by: 28)))
        s3 = vfmaq_f32(s3, q7, vld1q_f32(k3.advanced(by: 28)))

        let p01 = vpaddq_f32(s0, s1)
        let p23 = vpaddq_f32(s2, s3)
        let sum4 = vpaddq_f32(p01, p23)
        vst1q_f32(scores.advanced(by: tk), sum4)
        tk += 4
    }
    while tk < seqLen {
        let k0 = kHead.advanced(by: tk * 32)
        var s0 = vaddq_f32(vmulq_f32(q0, vld1q_f32(k0)), vmulq_f32(q1, vld1q_f32(k0.advanced(by: 4))))
        s0 = vfmaq_f32(s0, q2, vld1q_f32(k0.advanced(by: 8)))
        s0 = vfmaq_f32(s0, q3, vld1q_f32(k0.advanced(by: 12)))
        s0 = vfmaq_f32(s0, q4, vld1q_f32(k0.advanced(by: 16)))
        s0 = vfmaq_f32(s0, q5, vld1q_f32(k0.advanced(by: 20)))
        s0 = vfmaq_f32(s0, q6, vld1q_f32(k0.advanced(by: 24)))
        s0 = vfmaq_f32(s0, q7, vld1q_f32(k0.advanced(by: 28)))
        scores[tk] = vaddvq_f32(s0)
        tk += 1
    }
    #else
    for tk in 0..<seqLen {
        scores[tk] = dot32F32(a: qSlice, b: kHead.advanced(by: tk * 32)) * scale
    }
    #endif
}

@inlinable
public func softmaxInplace(scores: UnsafeMutablePointer<Float>, len: Int) {
    if len == 0 { return }
    #if arch(arm64)
    var vmax0 = vdupq_n_f32(-Float.infinity)
    var vmax1 = vdupq_n_f32(-Float.infinity)
    var i = 0
    while i + 16 <= len {
        let m0 = vmaxq_f32(vld1q_f32(scores.advanced(by: i)), vld1q_f32(scores.advanced(by: i + 4)))
        let m1 = vmaxq_f32(vld1q_f32(scores.advanced(by: i + 8)), vld1q_f32(scores.advanced(by: i + 12)))
        vmax0 = vmaxq_f32(vmax0, m0)
        vmax1 = vmaxq_f32(vmax1, m1)
        i += 16
    }
    var vmaxV = vmaxq_f32(vmax0, vmax1)
    while i + 4 <= len {
        vmaxV = vmaxq_f32(vmaxV, vld1q_f32(scores.advanced(by: i)))
        i += 4
    }
    var maxVal = vmaxvq_f32(vmaxV)
    while i < len {
        if scores[i] > maxVal {
            maxVal = scores[i]
        }
        i += 1
    }

    let vmax = vdupq_n_f32(maxVal)
    var vsum0 = vdupq_n_f32(0.0)
    var vsum1 = vdupq_n_f32(0.0)
    i = 0
    while i + 16 <= len {
        let e0 = neonExpF32(vsubq_f32(vld1q_f32(scores.advanced(by: i)), vmax))
        let e1 = neonExpF32(vsubq_f32(vld1q_f32(scores.advanced(by: i + 4)), vmax))
        let e2 = neonExpF32(vsubq_f32(vld1q_f32(scores.advanced(by: i + 8)), vmax))
        let e3 = neonExpF32(vsubq_f32(vld1q_f32(scores.advanced(by: i + 12)), vmax))
        vst1q_f32(scores.advanced(by: i), e0)
        vst1q_f32(scores.advanced(by: i + 4), e1)
        vst1q_f32(scores.advanced(by: i + 8), e2)
        vst1q_f32(scores.advanced(by: i + 12), e3)
        vsum0 = vaddq_f32(vsum0, vaddq_f32(e0, e1))
        vsum1 = vaddq_f32(vsum1, vaddq_f32(e2, e3))
        i += 16
    }
    var vsum = vaddq_f32(vsum0, vsum1)
    while i + 4 <= len {
        let e = neonExpF32(vsubq_f32(vld1q_f32(scores.advanced(by: i)), vmax))
        vst1q_f32(scores.advanced(by: i), e)
        vsum = vaddq_f32(vsum, e)
        i += 4
    }
    var sum = vaddvq_f32(vsum)
    while i < len {
        let e = Foundation.exp(scores[i] - maxVal)
        scores[i] = e
        sum += e
        i += 1
    }

    let invSum = 1.0 / sum
    let vinv = vdupq_n_f32(invSum)
    i = 0
    while i + 16 <= len {
        vst1q_f32(scores.advanced(by: i), vmulq_f32(vld1q_f32(scores.advanced(by: i)), vinv))
        vst1q_f32(scores.advanced(by: i + 4), vmulq_f32(vld1q_f32(scores.advanced(by: i + 4)), vinv))
        vst1q_f32(scores.advanced(by: i + 8), vmulq_f32(vld1q_f32(scores.advanced(by: i + 8)), vinv))
        vst1q_f32(scores.advanced(by: i + 12), vmulq_f32(vld1q_f32(scores.advanced(by: i + 12)), vinv))
        i += 16
    }
    while i + 4 <= len {
        vst1q_f32(scores.advanced(by: i), vmulq_f32(vld1q_f32(scores.advanced(by: i)), vinv))
        i += 4
    }
    while i < len {
        scores[i] *= invSum
        i += 1
    }
    #else
    var maxVal = -Float.infinity
    for i in 0..<len {
        if scores[i] > maxVal {
            maxVal = scores[i]
        }
    }
    var sum: Float = 0.0
    for i in 0..<len {
        let e = Foundation.exp(scores[i] - maxVal)
        scores[i] = e
        sum += e
    }
    let invSum = 1.0 / sum
    for i in 0..<len {
        scores[i] *= invSum
    }
    #endif
}

@inlinable
public func weightedSumHead32(
    probs: UnsafePointer<Float>,
    vHead: UnsafePointer<Float>,
    seqLen: Int,
    out32: UnsafeMutablePointer<Float>
) {
    #if arch(arm64)
    var acc0 = vdupq_n_f32(0.0)
    var acc1 = vdupq_n_f32(0.0)
    var acc2 = vdupq_n_f32(0.0)
    var acc3 = vdupq_n_f32(0.0)
    var acc4 = vdupq_n_f32(0.0)
    var acc5 = vdupq_n_f32(0.0)
    var acc6 = vdupq_n_f32(0.0)
    var acc7 = vdupq_n_f32(0.0)

    var t = 0
    while t + 4 <= seqLen {
        let vp = vld1q_f32(probs.advanced(by: t))
        let p0 = float32x4_t(repeating: vp[0])
        let p1 = float32x4_t(repeating: vp[1])
        let p2 = float32x4_t(repeating: vp[2])
        let p3 = float32x4_t(repeating: vp[3])

        let r0 = vHead.advanced(by: t * 32)
        let r1 = r0.advanced(by: 32)
        let r2 = r0.advanced(by: 64)
        let r3 = r0.advanced(by: 96)

        acc0 = vfmaq_f32(acc0, vld1q_f32(r0), p0)
        acc1 = vfmaq_f32(acc1, vld1q_f32(r0.advanced(by: 4)), p0)
        acc2 = vfmaq_f32(acc2, vld1q_f32(r0.advanced(by: 8)), p0)
        acc3 = vfmaq_f32(acc3, vld1q_f32(r0.advanced(by: 12)), p0)
        acc4 = vfmaq_f32(acc4, vld1q_f32(r0.advanced(by: 16)), p0)
        acc5 = vfmaq_f32(acc5, vld1q_f32(r0.advanced(by: 20)), p0)
        acc6 = vfmaq_f32(acc6, vld1q_f32(r0.advanced(by: 24)), p0)
        acc7 = vfmaq_f32(acc7, vld1q_f32(r0.advanced(by: 28)), p0)

        acc0 = vfmaq_f32(acc0, vld1q_f32(r1), p1)
        acc1 = vfmaq_f32(acc1, vld1q_f32(r1.advanced(by: 4)), p1)
        acc2 = vfmaq_f32(acc2, vld1q_f32(r1.advanced(by: 8)), p1)
        acc3 = vfmaq_f32(acc3, vld1q_f32(r1.advanced(by: 12)), p1)
        acc4 = vfmaq_f32(acc4, vld1q_f32(r1.advanced(by: 16)), p1)
        acc5 = vfmaq_f32(acc5, vld1q_f32(r1.advanced(by: 20)), p1)
        acc6 = vfmaq_f32(acc6, vld1q_f32(r1.advanced(by: 24)), p1)
        acc7 = vfmaq_f32(acc7, vld1q_f32(r1.advanced(by: 28)), p1)

        acc0 = vfmaq_f32(acc0, vld1q_f32(r2), p2)
        acc1 = vfmaq_f32(acc1, vld1q_f32(r2.advanced(by: 4)), p2)
        acc2 = vfmaq_f32(acc2, vld1q_f32(r2.advanced(by: 8)), p2)
        acc3 = vfmaq_f32(acc3, vld1q_f32(r2.advanced(by: 12)), p2)
        acc4 = vfmaq_f32(acc4, vld1q_f32(r2.advanced(by: 16)), p2)
        acc5 = vfmaq_f32(acc5, vld1q_f32(r2.advanced(by: 20)), p2)
        acc6 = vfmaq_f32(acc6, vld1q_f32(r2.advanced(by: 24)), p2)
        acc7 = vfmaq_f32(acc7, vld1q_f32(r2.advanced(by: 28)), p2)

        acc0 = vfmaq_f32(acc0, vld1q_f32(r3), p3)
        acc1 = vfmaq_f32(acc1, vld1q_f32(r3.advanced(by: 4)), p3)
        acc2 = vfmaq_f32(acc2, vld1q_f32(r3.advanced(by: 8)), p3)
        acc3 = vfmaq_f32(acc3, vld1q_f32(r3.advanced(by: 12)), p3)
        acc4 = vfmaq_f32(acc4, vld1q_f32(r3.advanced(by: 16)), p3)
        acc5 = vfmaq_f32(acc5, vld1q_f32(r3.advanced(by: 20)), p3)
        acc6 = vfmaq_f32(acc6, vld1q_f32(r3.advanced(by: 24)), p3)
        acc7 = vfmaq_f32(acc7, vld1q_f32(r3.advanced(by: 28)), p3)

        t += 4
    }
    while t < seqLen {
        let vp = vdupq_n_f32(probs[t])
        let row = vHead.advanced(by: t * 32)
        acc0 = vfmaq_f32(acc0, vld1q_f32(row), vp)
        acc1 = vfmaq_f32(acc1, vld1q_f32(row.advanced(by: 4)), vp)
        acc2 = vfmaq_f32(acc2, vld1q_f32(row.advanced(by: 8)), vp)
        acc3 = vfmaq_f32(acc3, vld1q_f32(row.advanced(by: 12)), vp)
        acc4 = vfmaq_f32(acc4, vld1q_f32(row.advanced(by: 16)), vp)
        acc5 = vfmaq_f32(acc5, vld1q_f32(row.advanced(by: 20)), vp)
        acc6 = vfmaq_f32(acc6, vld1q_f32(row.advanced(by: 24)), vp)
        acc7 = vfmaq_f32(acc7, vld1q_f32(row.advanced(by: 28)), vp)
        t += 1
    }
    vst1q_f32(out32, acc0)
    vst1q_f32(out32.advanced(by: 4), acc1)
    vst1q_f32(out32.advanced(by: 8), acc2)
    vst1q_f32(out32.advanced(by: 12), acc3)
    vst1q_f32(out32.advanced(by: 16), acc4)
    vst1q_f32(out32.advanced(by: 20), acc5)
    vst1q_f32(out32.advanced(by: 24), acc6)
    vst1q_f32(out32.advanced(by: 28), acc7)
    #else
    for d in 0..<32 {
        out32[d] = 0.0
    }
    for t in 0..<seqLen {
        let p = probs[t]
        let row = vHead.advanced(by: t * 32)
        for d in 0..<32 {
            out32[d] += p * row[d]
        }
    }
    #endif
}

@inlinable
public func dequantAddEmbeddingsRow(
    wRow: UnsafePointer<Int8>,
    wS: Float,
    pRow: UnsafePointer<Int8>,
    pS: Float,
    tRow: UnsafePointer<Float>,
    hRow: UnsafeMutablePointer<Float>,
    count n: Int
) {
    #if arch(arm64)
    let vws = vdupq_n_f32(wS)
    let vps = vdupq_n_f32(pS)
    var i = 0
    while i + 16 <= n {
        let vw8 = vld1q_s8(wRow.advanced(by: i))
        let vp8 = vld1q_s8(pRow.advanced(by: i))
        let vwLo16 = vmovl_s8(vget_low_s8(vw8))
        let vwHi16 = vmovl_s8(vget_high_s8(vw8))
        let vpLo16 = vmovl_s8(vget_low_s8(vp8))
        let vpHi16 = vmovl_s8(vget_high_s8(vp8))

        let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vwLo16)))
        let w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vwLo16)))
        let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vwHi16)))
        let w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vwHi16)))

        let p0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vpLo16)))
        let p1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vpLo16)))
        let p2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vpHi16)))
        let p3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vpHi16)))

        let r0 = vfmaq_f32(vfmaq_f32(vld1q_f32(tRow.advanced(by: i)), w0, vws), p0, vps)
        let r1 = vfmaq_f32(vfmaq_f32(vld1q_f32(tRow.advanced(by: i + 4)), w1, vws), p1, vps)
        let r2 = vfmaq_f32(vfmaq_f32(vld1q_f32(tRow.advanced(by: i + 8)), w2, vws), p2, vps)
        let r3 = vfmaq_f32(vfmaq_f32(vld1q_f32(tRow.advanced(by: i + 12)), w3, vws), p3, vps)

        vst1q_f32(hRow.advanced(by: i), r0)
        vst1q_f32(hRow.advanced(by: i + 4), r1)
        vst1q_f32(hRow.advanced(by: i + 8), r2)
        vst1q_f32(hRow.advanced(by: i + 12), r3)
        i += 16
    }
    #else
    for j in 0..<n {
        hRow[j] = Float(wRow[j]) * wS + Float(pRow[j]) * pS + tRow[j]
    }
    #endif
}

// MARK: - L1-Tiled Full-Int8 Multi-Head Self-Attention (FEAT_SME2)

#if arch(arm64)
public func attentionMultiheadSme(
    seqLen: Int,
    qkvBuf: UnsafePointer<Float>,
    qTF32: UnsafeMutablePointer<Float>,
    kTF32: UnsafeMutablePointer<Float>,
    vContigF32: UnsafeMutablePointer<Float>,
    scoresBlk: UnsafeMutablePointer<Float>,
    pTF32: UnsafeMutablePointer<Float>,
    attnCtx: UnsafeMutablePointer<Float>,
    segRanges: UnsafeBufferPointer<(Int, Int)>?
) {
    attentionHeadsSme(
        seqLen: seqLen,
        qkvBuf: qkvBuf,
        qTF32: qTF32,
        kTF32: kTF32,
        vContigF32: vContigF32,
        scoresBlk: scoresBlk,
        pTF32: pTF32,
        attnCtx: attnCtx,
        segRanges: segRanges,
        heads: 0..<12
    )
}

public func attentionHeadsSme(
    seqLen: Int,
    qkvBuf: UnsafePointer<Float>,
    qTF32: UnsafeMutablePointer<Float>,
    kTF32: UnsafeMutablePointer<Float>,
    vContigF32: UnsafeMutablePointer<Float>,
    scoresBlk: UnsafeMutablePointer<Float>,
    pTF32: UnsafeMutablePointer<Float>,
    attnCtx: UnsafeMutablePointer<Float>,
    segRanges: UnsafeBufferPointer<(Int, Int)>?,
    heads: Range<Int>
) {
    let HIDDEN = 384
    let HDIM = 32
    let invSqrtD: Float = 1.0 / Float(HDIM).squareRoot()
    let mBlocks = (seqLen + 15) / 16
    let kSuper = (mBlocks + 3) / 4
    let seqStride = kSuper * 64
    let tkSteps = (seqLen + 1) / 2
    let tkStepsAligned = ((tkSteps + 3) / 4) * 4
    let tkAligned = tkStepsAligned * 2

    let qTI8 = UnsafeMutableRawPointer(qTF32).assumingMemoryBound(to: Int8.self)
    let kTI8 = UnsafeMutableRawPointer(kTF32).assumingMemoryBound(to: Int8.self)
    let qScale = qTF32.advanced(by: 8192)
    let kScale = kTF32.advanced(by: 8192)

    let vPackedI16 = UnsafeMutableRawPointer(vContigF32).assumingMemoryBound(to: Int16.self)
    let vScale = vContigF32.advanced(by: 8192)

    let probsBlkI16 = UnsafeMutableRawPointer(pTF32).assumingMemoryBound(to: Int16.self)
    let pTI16 = UnsafeMutableRawPointer(pTF32).advanced(by: 16384).assumingMemoryBound(to: Int16.self)
    let pScale = pTF32.advanced(by: 8192)

    let kernels = AArch64AsmKernels.shared

    for h in heads {
        let hOff = h * HDIM
        packQkvHeadInt8Neon(
            qkv: qkvBuf,
            seqLen: seqLen,
            seqStride: seqStride,
            tkSteps: tkSteps,
            hOff: hOff,
            kScaleMul: invSqrtD * LOG2_E_F32,
            qTI8: qTI8,
            kTI8: kTI8,
            qScale: qScale,
            kScale: kScale,
            vPackedI16: vPackedI16,
            vScale: vScale
        )

        for qb in 0..<mBlocks {
            let qBase = qb * 16
            let validQ = min(seqLen - qBase, 16)
            let pqBlk = UnsafePointer(qTI8.advanced(by: qb * 512))
            let pk0Init = UnsafePointer(kTI8)

            // 1. Q_blk @ K_T using 4-tile Int8 smopa (8 K-steps)
            kernels.attentionQktSme(pqBlk, pk0Init, scoresBlk, seqStride * 4, kSuper, validQ)

            // 2. Two-Pass Exact-Range Int16 Softmax
            if let ranges = segRanges, let rangesBase = ranges.baseAddress {
                for r in 0..<validQ {
                    let (sStart, sEnd) = rangesBase[qBase + r]
                    let sRow = scoresBlk.advanced(by: r * seqStride)
                    let pRow = probsBlkI16.advanced(by: r * seqStride)
                    for z in 0..<sStart {
                        pRow[z] = 0
                    }
                    pScale[r] = softmaxQuantizeI16FromAccNeon(
                        row: sRow.advanced(by: sStart),
                        len: sEnd - sStart,
                        kScale: UnsafePointer(kScale.advanced(by: sStart)),
                        qScale: qScale[qBase + r],
                        outI16: pRow.advanced(by: sStart)
                    )
                    for z in sEnd..<tkAligned {
                        pRow[z] = 0
                    }
                }
            } else {
                for r in 0..<validQ {
                    let sRow = scoresBlk.advanced(by: r * seqStride)
                    let pRow = probsBlkI16.advanced(by: r * seqStride)
                    pScale[r] = softmaxQuantizeI16FromAccNeon(
                        row: sRow,
                        len: seqLen,
                        kScale: UnsafePointer(kScale),
                        qScale: qScale[qBase + r],
                        outI16: pRow
                    )
                    for z in seqLen..<tkAligned {
                        pRow[z] = 0
                    }
                }
            }

            // 3. Transpose 4-byte groups (2 keys of i16) of `probsBlkI16` into `pTI16`
            transposeProbs16I8Neon(
                probsBlkI8: UnsafeRawPointer(probsBlkI16).assumingMemoryBound(to: Int8.self),
                seqStride: seqStride * 2,
                validQ: validQ,
                tkStepsAligned: tkStepsAligned,
                pTI8: UnsafeMutableRawPointer(pTI16).assumingMemoryBound(to: Int8.self)
            )

            // 4. P_T_i16 @ V_packed_i16 with 2-way int16 smopa (za0, za1)
            let pctxOut = attnCtx.advanced(by: qBase * HIDDEN + hOff)
            kernels.attentionPvSme(
                UnsafePointer(pTI16),
                UnsafePointer(vPackedI16),
                UnsafePointer(vScale),
                UnsafePointer(pScale),
                tkSteps,
                pctxOut,
                validQ
            )
        }
    }
}

private let P_Q16_MAX: Float = 16383.0

@usableFromInline
@inline(__always)
func exp2QNeon(_ xIn: float32x4_t, scale: Float, xMin: Float) -> int32x4_t {
    let x = vmaxq_f32(xIn, vdupq_n_f32(xMin))
    let n = vcvtmq_s32_f32(x)
    let f = vsubq_f32(x, vcvtq_f32_s32(n))
    var p = vfmaq_f32(
        vdupq_n_f32(0.225_119_37 * scale),
        f,
        vdupq_n_f32(0.078_421_69 * scale)
    )
    p = vfmaq_f32(vdupq_n_f32(0.696_385_3 * scale), f, p)
    p = vfmaq_f32(vdupq_n_f32(0.999_857 * scale), f, p)
    let scaled = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), n &<< 23))
    return vcvtnq_s32_f32(scaled)
}

@usableFromInline
@inline(__always)
func softmaxQuantizeI16FromAccNeon(
    row: UnsafeMutablePointer<Float>,
    len: Int,
    kScale: UnsafePointer<Float>,
    qScale: Float,
    outI16: UnsafeMutablePointer<Int16>
) -> Float {
    if len == 0 { return 0.0 }
    let acc = UnsafeRawPointer(row).assumingMemoryBound(to: Int32.self)
    var vmax0 = vdupq_n_f32(-Float.infinity)
    var vmax1 = vdupq_n_f32(-Float.infinity)
    let vq = vdupq_n_f32(qScale)
    var i = 0
    while i + 8 <= len {
        let s0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(acc.advanced(by: i))), vld1q_f32(kScale.advanced(by: i))), vq)
        let s1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(acc.advanced(by: i + 4))), vld1q_f32(kScale.advanced(by: i + 4))), vq)
        vst1q_f32(row.advanced(by: i), s0)
        vst1q_f32(row.advanced(by: i + 4), s1)
        vmax0 = vmaxq_f32(vmax0, s0)
        vmax1 = vmaxq_f32(vmax1, s1)
        i += 8
    }
    var maxVal = vmaxvq_f32(vmaxq_f32(vmax0, vmax1))
    while i < len {
        let v = Float(acc[i]) * kScale[i] * qScale
        row[i] = v
        if v > maxVal {
            maxVal = v
        }
        i += 1
    }

    let vmax = vdupq_n_f32(maxVal)
    var vsum0 = vdupq_n_s32(0)
    var vsum1 = vdupq_n_s32(0)
    i = 0
    while i + 8 <= len {
        let q0 = exp2QNeon(vsubq_f32(vld1q_f32(row.advanced(by: i)), vmax), scale: P_Q16_MAX, xMin: -16.0)
        let q1 = exp2QNeon(vsubq_f32(vld1q_f32(row.advanced(by: i + 4)), vmax), scale: P_Q16_MAX, xMin: -16.0)
        vsum0 = vaddq_s32(vsum0, q0)
        vsum1 = vaddq_s32(vsum1, q1)
        vst1q_s16(outI16.advanced(by: i), vqmovn_high_s32(vqmovn_s32(q0), q1))
        i += 8
    }
    var isum = vaddvq_s32(vaddq_s32(vsum0, vsum1))
    while i < len {
        let e = Int32((Foundation.exp2(row[i] - maxVal) * P_Q16_MAX).rounded(.toNearestOrAwayFromZero))
        let q = min(max(e, 0), Int32(P_Q16_MAX))
        outI16[i] = Int16(truncatingIfNeeded: q)
        isum &+= q
        i += 1
    }
    return 1.0 / Float(max(isum, 1))
}

@usableFromInline
@inline(__always)
func transposeProbs16I8Neon(
    probsBlkI8: UnsafePointer<Int8>,
    seqStride: Int,
    validQ: Int,
    tkStepsAligned: Int,
    pTI8: UnsafeMutablePointer<Int8>
) {
    let zero = vdupq_n_s32(0)
    let srcI32 = UnsafeRawPointer(probsBlkI8).assumingMemoryBound(to: Int32.self)
    let dstI32 = UnsafeMutableRawPointer(pTI8).assumingMemoryBound(to: Int32.self)
    let strideI32 = seqStride / 4

    var ks = 0
    while ks + 4 <= tkStepsAligned {
        let dst = dstI32.advanced(by: ks * 16)
        for rg in 0..<4 {
            let r0 = rg * 4
            if r0 >= validQ {
                vst1q_s32(dst.advanced(by: r0), zero)
                vst1q_s32(dst.advanced(by: 16 + r0), zero)
                vst1q_s32(dst.advanced(by: 32 + r0), zero)
                vst1q_s32(dst.advanced(by: 48 + r0), zero)
                continue
            }
            let v0 = vld1q_s32(srcI32.advanced(by: r0 * strideI32 + ks))
            let v1 = (r0 + 1 < validQ) ? vld1q_s32(srcI32.advanced(by: (r0 + 1) * strideI32 + ks)) : zero
            let v2 = (r0 + 2 < validQ) ? vld1q_s32(srcI32.advanced(by: (r0 + 2) * strideI32 + ks)) : zero
            let v3 = (r0 + 3 < validQ) ? vld1q_s32(srcI32.advanced(by: (r0 + 3) * strideI32 + ks)) : zero

            let t0 = vtrn1q_s32(v0, v1)
            let t1 = vtrn2q_s32(v0, v1)
            let t2 = vtrn1q_s32(v2, v3)
            let t3 = vtrn2q_s32(v2, v3)

            vst1q_s32(dst.advanced(by: r0), vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)))
            vst1q_s32(dst.advanced(by: 16 + r0), vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)))
            vst1q_s32(dst.advanced(by: 32 + r0), vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)))
            vst1q_s32(dst.advanced(by: 48 + r0), vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)))
        }
        ks += 4
    }
}

@usableFromInline
@inline(__always)
func quantizeHead32RowNeon(
    src: UnsafePointer<Float>,
    outScaleMul: Float
) -> (int32x4_t, int32x4_t, Float) {
    let v0 = vld1q_f32(src)
    let v1 = vld1q_f32(src.advanced(by: 4))
    let v2 = vld1q_f32(src.advanced(by: 8))
    let v3 = vld1q_f32(src.advanced(by: 12))
    let v4 = vld1q_f32(src.advanced(by: 16))
    let v5 = vld1q_f32(src.advanced(by: 20))
    let v6 = vld1q_f32(src.advanced(by: 24))
    let v7 = vld1q_f32(src.advanced(by: 28))

    let m01 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1))
    let m23 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3))
    let m45 = vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5))
    let m67 = vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7))
    let m03 = vmaxq_f32(m01, m23)
    let m47 = vmaxq_f32(m45, m67)
    let maxAbs = max(vmaxvq_f32(vmaxq_f32(m03, m47)), 1e-8)

    let invS = vdupq_n_f32(127.0 / maxAbs)
    let q0 = vcvtnq_s32_f32(vmulq_f32(v0, invS))
    let q1 = vcvtnq_s32_f32(vmulq_f32(v1, invS))
    let q2 = vcvtnq_s32_f32(vmulq_f32(v2, invS))
    let q3 = vcvtnq_s32_f32(vmulq_f32(v3, invS))
    let q4 = vcvtnq_s32_f32(vmulq_f32(v4, invS))
    let q5 = vcvtnq_s32_f32(vmulq_f32(v5, invS))
    let q6 = vcvtnq_s32_f32(vmulq_f32(v6, invS))
    let q7 = vcvtnq_s32_f32(vmulq_f32(v7, invS))

    let lo16a = vqmovn_high_s32(vqmovn_s32(q0), q1)
    let lo16b = vqmovn_high_s32(vqmovn_s32(q2), q3)
    let loI8 = vqmovn_high_s16(vqmovn_s16(lo16a), lo16b)

    let hi16a = vqmovn_high_s32(vqmovn_s32(q4), q5)
    let hi16b = vqmovn_high_s32(vqmovn_s32(q6), q7)
    let hiI8 = vqmovn_high_s16(vqmovn_s16(hi16a), hi16b)

    return (
        vreinterpretq_s32_s8(loI8),
        vreinterpretq_s32_s8(hiI8),
        (maxAbs / 127.0) * outScaleMul
    )
}

@usableFromInline
@inline(__always)
func quantizeVRowNeon(
    src: UnsafePointer<Float>,
    inv0: float32x4_t,
    inv1: float32x4_t,
    inv2: float32x4_t,
    inv3: float32x4_t,
    inv4: float32x4_t,
    inv5: float32x4_t,
    inv6: float32x4_t,
    inv7: float32x4_t
) -> (int8x16_t, int8x16_t) {
    let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src), inv0))
    let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.advanced(by: 4)), inv1))
    let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.advanced(by: 8)), inv2))
    let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.advanced(by: 12)), inv3))
    let q4 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.advanced(by: 16)), inv4))
    let q5 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.advanced(by: 20)), inv5))
    let q6 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.advanced(by: 24)), inv6))
    let q7 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.advanced(by: 28)), inv7))

    let lo16a = vqmovn_high_s32(vqmovn_s32(q0), q1)
    let lo16b = vqmovn_high_s32(vqmovn_s32(q2), q3)
    let loI8 = vqmovn_high_s16(vqmovn_s16(lo16a), lo16b)

    let hi16a = vqmovn_high_s32(vqmovn_s32(q4), q5)
    let hi16b = vqmovn_high_s32(vqmovn_s32(q6), q7)
    let hiI8 = vqmovn_high_s16(vqmovn_s16(hi16a), hi16b)

    return (loI8, hiI8)
}

@usableFromInline
func packQkvHeadInt8Neon(
    qkv: UnsafePointer<Float>,
    seqLen: Int,
    seqStride: Int,
    tkSteps: Int,
    hOff: Int,
    kScaleMul: Float,
    qTI8: UnsafeMutablePointer<Int8>,
    kTI8: UnsafeMutablePointer<Int8>,
    qScale: UnsafeMutablePointer<Float>,
    kScale: UnsafeMutablePointer<Float>,
    vPackedI16: UnsafeMutablePointer<Int16>,
    vScale: UnsafeMutablePointer<Float>
) {
    let HIDDEN = 384
    let mBlocks = (seqLen + 15) / 16
    let stride = 3 * HIDDEN
    let zeroI32 = vdupq_n_s32(0)
    let zeroI8 = vdupq_n_s8(0)

    var vm0 = vdupq_n_f32(1e-8)
    var vm1 = vdupq_n_f32(1e-8)
    var vm2 = vdupq_n_f32(1e-8)
    var vm3 = vdupq_n_f32(1e-8)
    var vm4 = vdupq_n_f32(1e-8)
    var vm5 = vdupq_n_f32(1e-8)
    var vm6 = vdupq_n_f32(1e-8)
    var vm7 = vdupq_n_f32(1e-8)

    for t in 0..<seqLen {
        let src = qkv.advanced(by: t * stride + 2 * HIDDEN + hOff)
        vm0 = vmaxq_f32(vm0, vabsq_f32(vld1q_f32(src)))
        vm1 = vmaxq_f32(vm1, vabsq_f32(vld1q_f32(src.advanced(by: 4))))
        vm2 = vmaxq_f32(vm2, vabsq_f32(vld1q_f32(src.advanced(by: 8))))
        vm3 = vmaxq_f32(vm3, vabsq_f32(vld1q_f32(src.advanced(by: 12))))
        vm4 = vmaxq_f32(vm4, vabsq_f32(vld1q_f32(src.advanced(by: 16))))
        vm5 = vmaxq_f32(vm5, vabsq_f32(vld1q_f32(src.advanced(by: 20))))
        vm6 = vmaxq_f32(vm6, vabsq_f32(vld1q_f32(src.advanced(by: 24))))
        vm7 = vmaxq_f32(vm7, vabsq_f32(vld1q_f32(src.advanced(by: 28))))
    }

    let vInv127 = vdupq_n_f32(1.0 / 127.0)
    let v127 = vdupq_n_f32(127.0)
    vst1q_f32(vScale, vmulq_f32(vm0, vInv127))
    vst1q_f32(vScale.advanced(by: 4), vmulq_f32(vm1, vInv127))
    vst1q_f32(vScale.advanced(by: 8), vmulq_f32(vm2, vInv127))
    vst1q_f32(vScale.advanced(by: 12), vmulq_f32(vm3, vInv127))
    vst1q_f32(vScale.advanced(by: 16), vmulq_f32(vm4, vInv127))
    vst1q_f32(vScale.advanced(by: 20), vmulq_f32(vm5, vInv127))
    vst1q_f32(vScale.advanced(by: 24), vmulq_f32(vm6, vInv127))
    vst1q_f32(vScale.advanced(by: 28), vmulq_f32(vm7, vInv127))

    let inv0 = vdivq_f32(v127, vm0)
    let inv1 = vdivq_f32(v127, vm1)
    let inv2 = vdivq_f32(v127, vm2)
    let inv3 = vdivq_f32(v127, vm3)
    let inv4 = vdivq_f32(v127, vm4)
    let inv5 = vdivq_f32(v127, vm5)
    let inv6 = vdivq_f32(v127, vm6)
    let inv7 = vdivq_f32(v127, vm7)

    for ks in 0..<tkSteps {
        let t0 = ks * 2
        let dst = vPackedI16.advanced(by: ks * 64)
        let (r0Lo, r0Hi) = quantizeVRowNeon(
            src: qkv.advanced(by: t0 * stride + 2 * HIDDEN + hOff),
            inv0: inv0, inv1: inv1, inv2: inv2, inv3: inv3,
            inv4: inv4, inv5: inv5, inv6: inv6, inv7: inv7
        )
        let (r1Lo, r1Hi): (int8x16_t, int8x16_t)
        if t0 + 1 < seqLen {
            (r1Lo, r1Hi) = quantizeVRowNeon(
                src: qkv.advanced(by: (t0 + 1) * stride + 2 * HIDDEN + hOff),
                inv0: inv0, inv1: inv1, inv2: inv2, inv3: inv3,
                inv4: inv4, inv5: inv5, inv6: inv6, inv7: inv7
            )
        } else {
            (r1Lo, r1Hi) = (zeroI8, zeroI8)
        }
        let a0 = vmovl_s8(vget_low_s8(r0Lo))
        let a1 = vmovl_high_s8(r0Lo)
        let a2 = vmovl_s8(vget_low_s8(r0Hi))
        let a3 = vmovl_high_s8(r0Hi)

        let b0 = vmovl_s8(vget_low_s8(r1Lo))
        let b1 = vmovl_high_s8(r1Lo)
        let b2 = vmovl_s8(vget_low_s8(r1Hi))
        let b3 = vmovl_high_s8(r1Hi)

        vst1q_s16(dst, vzip1q_s16(a0, b0))
        vst1q_s16(dst.advanced(by: 8), vzip2q_s16(a0, b0))
        vst1q_s16(dst.advanced(by: 16), vzip1q_s16(a1, b1))
        vst1q_s16(dst.advanced(by: 24), vzip2q_s16(a1, b1))
        vst1q_s16(dst.advanced(by: 32), vzip1q_s16(a2, b2))
        vst1q_s16(dst.advanced(by: 40), vzip2q_s16(a2, b2))
        vst1q_s16(dst.advanced(by: 48), vzip1q_s16(a3, b3))
        vst1q_s16(dst.advanced(by: 56), vzip2q_s16(a3, b3))
    }

    let qTI32 = UnsafeMutableRawPointer(qTI8).assumingMemoryBound(to: Int32.self)
    let kTI32 = UnsafeMutableRawPointer(kTI8).assumingMemoryBound(to: Int32.self)

    for mb in 0..<mBlocks {
        let tBase = mb * 16
        let valid = min(seqLen - tBase, 16)
        let qBlk = qTI32.advanced(by: mb * 128)
        let kBlk = kTI32.advanced(by: mb * 128)

        for rg in 0..<4 {
            let r0 = rg * 4
            let (q0Lo, q0Hi, qs0) = (r0 < valid)
                ? quantizeHead32RowNeon(src: qkv.advanced(by: (tBase + r0) * stride + hOff), outScaleMul: 1.0)
                : (zeroI32, zeroI32, Float(0.0))
            let (q1Lo, q1Hi, qs1) = (r0 + 1 < valid)
                ? quantizeHead32RowNeon(src: qkv.advanced(by: (tBase + r0 + 1) * stride + hOff), outScaleMul: 1.0)
                : (zeroI32, zeroI32, Float(0.0))
            let (q2Lo, q2Hi, qs2) = (r0 + 2 < valid)
                ? quantizeHead32RowNeon(src: qkv.advanced(by: (tBase + r0 + 2) * stride + hOff), outScaleMul: 1.0)
                : (zeroI32, zeroI32, Float(0.0))
            let (q3Lo, q3Hi, qs3) = (r0 + 3 < valid)
                ? quantizeHead32RowNeon(src: qkv.advanced(by: (tBase + r0 + 3) * stride + hOff), outScaleMul: 1.0)
                : (zeroI32, zeroI32, Float(0.0))

            qScale[tBase + r0] = qs0
            qScale[tBase + r0 + 1] = qs1
            qScale[tBase + r0 + 2] = qs2
            qScale[tBase + r0 + 3] = qs3

            let qt0 = vtrn1q_s32(q0Lo, q1Lo)
            let qt1 = vtrn2q_s32(q0Lo, q1Lo)
            let qt2 = vtrn1q_s32(q2Lo, q3Lo)
            let qt3 = vtrn2q_s32(q2Lo, q3Lo)
            vst1q_s32(qBlk.advanced(by: r0), vcombine_s32(vget_low_s32(qt0), vget_low_s32(qt2)))
            vst1q_s32(qBlk.advanced(by: 16 + r0), vcombine_s32(vget_low_s32(qt1), vget_low_s32(qt3)))
            vst1q_s32(qBlk.advanced(by: 32 + r0), vcombine_s32(vget_high_s32(qt0), vget_high_s32(qt2)))
            vst1q_s32(qBlk.advanced(by: 48 + r0), vcombine_s32(vget_high_s32(qt1), vget_high_s32(qt3)))

            let qt4 = vtrn1q_s32(q0Hi, q1Hi)
            let qt5 = vtrn2q_s32(q0Hi, q1Hi)
            let qt6 = vtrn1q_s32(q2Hi, q3Hi)
            let qt7 = vtrn2q_s32(q2Hi, q3Hi)
            vst1q_s32(qBlk.advanced(by: 64 + r0), vcombine_s32(vget_low_s32(qt4), vget_low_s32(qt6)))
            vst1q_s32(qBlk.advanced(by: 80 + r0), vcombine_s32(vget_low_s32(qt5), vget_low_s32(qt7)))
            vst1q_s32(qBlk.advanced(by: 96 + r0), vcombine_s32(vget_high_s32(qt4), vget_high_s32(qt6)))
            vst1q_s32(qBlk.advanced(by: 112 + r0), vcombine_s32(vget_high_s32(qt5), vget_high_s32(qt7)))

            let (k0Lo, k0Hi, ks0) = (r0 < valid)
                ? quantizeHead32RowNeon(src: qkv.advanced(by: (tBase + r0) * stride + HIDDEN + hOff), outScaleMul: kScaleMul)
                : (zeroI32, zeroI32, Float(0.0))
            let (k1Lo, k1Hi, ks1) = (r0 + 1 < valid)
                ? quantizeHead32RowNeon(src: qkv.advanced(by: (tBase + r0 + 1) * stride + HIDDEN + hOff), outScaleMul: kScaleMul)
                : (zeroI32, zeroI32, Float(0.0))
            let (k2Lo, k2Hi, ks2) = (r0 + 2 < valid)
                ? quantizeHead32RowNeon(src: qkv.advanced(by: (tBase + r0 + 2) * stride + HIDDEN + hOff), outScaleMul: kScaleMul)
                : (zeroI32, zeroI32, Float(0.0))
            let (k3Lo, k3Hi, ks3) = (r0 + 3 < valid)
                ? quantizeHead32RowNeon(src: qkv.advanced(by: (tBase + r0 + 3) * stride + HIDDEN + hOff), outScaleMul: kScaleMul)
                : (zeroI32, zeroI32, Float(0.0))

            kScale[tBase + r0] = ks0
            kScale[tBase + r0 + 1] = ks1
            kScale[tBase + r0 + 2] = ks2
            kScale[tBase + r0 + 3] = ks3

            let kt0 = vtrn1q_s32(k0Lo, k1Lo)
            let kt1 = vtrn2q_s32(k0Lo, k1Lo)
            let kt2 = vtrn1q_s32(k2Lo, k3Lo)
            let kt3 = vtrn2q_s32(k2Lo, k3Lo)
            vst1q_s32(kBlk.advanced(by: r0), vcombine_s32(vget_low_s32(kt0), vget_low_s32(kt2)))
            vst1q_s32(kBlk.advanced(by: 16 + r0), vcombine_s32(vget_low_s32(kt1), vget_low_s32(kt3)))
            vst1q_s32(kBlk.advanced(by: 32 + r0), vcombine_s32(vget_high_s32(kt0), vget_high_s32(kt2)))
            vst1q_s32(kBlk.advanced(by: 48 + r0), vcombine_s32(vget_high_s32(kt1), vget_high_s32(kt3)))

            let kt4 = vtrn1q_s32(k0Hi, k1Hi)
            let kt5 = vtrn2q_s32(k0Hi, k1Hi)
            let kt6 = vtrn1q_s32(k2Hi, k3Hi)
            let kt7 = vtrn2q_s32(k2Hi, k3Hi)
            vst1q_s32(kBlk.advanced(by: 64 + r0), vcombine_s32(vget_low_s32(kt4), vget_low_s32(kt6)))
            vst1q_s32(kBlk.advanced(by: 80 + r0), vcombine_s32(vget_low_s32(kt5), vget_low_s32(kt7)))
            vst1q_s32(kBlk.advanced(by: 96 + r0), vcombine_s32(vget_high_s32(kt4), vget_high_s32(kt6)))
            vst1q_s32(kBlk.advanced(by: 112 + r0), vcombine_s32(vget_high_s32(kt5), vget_high_s32(kt7)))
        }
    }

    if mBlocks * 16 < seqStride {
        for t in (mBlocks * 16)..<seqStride {
            kScale[t] = 0.0
        }
    }
}
#endif

@inlinable
@inline(__always)
public func extractKvHead32(
    qkvBuf: UnsafePointer<Float>,
    seqLen: Int,
    rowStride: Int,
    kColOff: Int,
    vColOff: Int,
    kHead: UnsafeMutablePointer<Float>,
    vHead: UnsafeMutablePointer<Float>
) {
    #if arch(arm64)
    for t in 0..<seqLen {
        let row = qkvBuf.advanced(by: t * rowStride)
        let ks = row.advanced(by: kColOff)
        let vs = row.advanced(by: vColOff)
        let kd = kHead.advanced(by: t * 32)
        let vd = vHead.advanced(by: t * 32)

        vst1q_f32(kd, vld1q_f32(ks))
        vst1q_f32(kd.advanced(by: 4), vld1q_f32(ks.advanced(by: 4)))
        vst1q_f32(kd.advanced(by: 8), vld1q_f32(ks.advanced(by: 8)))
        vst1q_f32(kd.advanced(by: 12), vld1q_f32(ks.advanced(by: 12)))
        vst1q_f32(kd.advanced(by: 16), vld1q_f32(ks.advanced(by: 16)))
        vst1q_f32(kd.advanced(by: 20), vld1q_f32(ks.advanced(by: 20)))
        vst1q_f32(kd.advanced(by: 24), vld1q_f32(ks.advanced(by: 24)))
        vst1q_f32(kd.advanced(by: 28), vld1q_f32(ks.advanced(by: 28)))

        vst1q_f32(vd, vld1q_f32(vs))
        vst1q_f32(vd.advanced(by: 4), vld1q_f32(vs.advanced(by: 4)))
        vst1q_f32(vd.advanced(by: 8), vld1q_f32(vs.advanced(by: 8)))
        vst1q_f32(vd.advanced(by: 12), vld1q_f32(vs.advanced(by: 12)))
        vst1q_f32(vd.advanced(by: 16), vld1q_f32(vs.advanced(by: 16)))
        vst1q_f32(vd.advanced(by: 20), vld1q_f32(vs.advanced(by: 20)))
        vst1q_f32(vd.advanced(by: 24), vld1q_f32(vs.advanced(by: 24)))
        vst1q_f32(vd.advanced(by: 28), vld1q_f32(vs.advanced(by: 28)))
    }
    #else
    let byteCount = 32 * MemoryLayout<Float>.stride
    for t in 0..<seqLen {
        let row = qkvBuf.advanced(by: t * rowStride)
        UnsafeMutableRawPointer(kHead.advanced(by: t * 32)).copyMemory(
            from: UnsafeRawPointer(row.advanced(by: kColOff)),
            byteCount: byteCount
        )
        UnsafeMutableRawPointer(vHead.advanced(by: t * 32)).copyMemory(
            from: UnsafeRawPointer(row.advanced(by: vColOff)),
            byteCount: byteCount
        )
    }
    #endif
}

@inlinable
public func l2NormalizeInplace(v: UnsafeMutablePointer<Float>, count n: Int) {
    var sumSq: Float = 0.0
    for i in 0..<n {
        let x = v[i]
        sumSq += x * x
    }
    if sumSq > 0.0 {
        let invNorm = 1.0 / sumSq.squareRoot()
        for i in 0..<n {
            v[i] *= invNorm
        }
    }
}
