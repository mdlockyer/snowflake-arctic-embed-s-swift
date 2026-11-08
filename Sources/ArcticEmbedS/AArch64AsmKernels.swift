// Auto-assembled AArch64 SME2 / I8MM / FP16 / DotProd machine-code kernels.
// Loaded once into an executable page at runtime when the host CPU supports the ISA.
#if arch(arm64)
import Foundation
#if canImport(Darwin)
import Darwin
@_silgen_name("sys_icache_invalidate")
private func sys_icache_invalidate(_ start: UnsafeMutableRawPointer, _ len: Int)
#elseif canImport(Glibc)
import Glibc
@_silgen_name("__clear_cache")
private func clear_cache(_ start: UnsafeMutableRawPointer, _ end: UnsafeMutableRawPointer)
#endif

@usableFromInline
final class AArch64AsmKernels: @unchecked Sendable {
    @usableFromInline static let shared = AArch64AsmKernels()

    @usableFromInline typealias GemmSmeRowblockFn = @convention(c) (
        UnsafePointer<Int8>, UnsafePointer<Int8>, UnsafePointer<Float>, UnsafePointer<Float>,
        UnsafePointer<Float>, UnsafePointer<Float>?, UnsafeMutablePointer<Float>, UInt64
    ) -> Void
    @usableFromInline typealias AttentionQktSmeFn = @convention(c) (
        UnsafePointer<Int8>, UnsafePointer<Int8>, UnsafeMutablePointer<Float>, Int, Int, Int
    ) -> Void
    @usableFromInline typealias AttentionPvSmeFn = @convention(c) (
        UnsafePointer<Int16>, UnsafePointer<Int16>, UnsafePointer<Float>, UnsafePointer<Float>,
        Int, UnsafeMutablePointer<Float>, Int
    ) -> Void
    @usableFromInline typealias F32ToF16NeonFn = @convention(c) (
        UnsafePointer<Float>, UnsafeMutablePointer<UInt16>, Int
    ) -> Void
    @usableFromInline typealias GeluF16Pass1Fn = @convention(c) (
        UnsafePointer<Float>, UnsafeMutablePointer<Float>, UnsafePointer<UInt16>, Int
    ) -> UInt32
    @usableFromInline typealias GeluF16Pass2Fn = @convention(c) (
        UInt32, UnsafePointer<UInt16>, UnsafeMutablePointer<Int8>, Int
    ) -> Void
    @usableFromInline typealias F16x8ToF32x8Fn = @convention(c) (
        UnsafePointer<UInt16>, UnsafeMutablePointer<Float>
    ) -> Void
    @usableFromInline typealias SmmlaTileFn = @convention(c) (
        UnsafePointer<Int8>, UnsafePointer<Int8>, Int, UnsafeMutablePointer<Int32>
    ) -> Void
    @usableFromInline typealias SmmlaM12PanelFn = @convention(c) (
        UnsafePointer<Int8>, UnsafePointer<Int8>, UnsafePointer<Int8>, Int, UnsafeMutablePointer<Int32>
    ) -> Void
    @usableFromInline typealias Sdot4RowPanelFn = @convention(c) (
        UnsafePointer<Int32>, UnsafePointer<Int32>, UnsafePointer<Int32>, UnsafePointer<Int32>,
        UnsafePointer<Int8>, Int, UnsafeMutablePointer<Int32>
    ) -> Void
    @usableFromInline typealias Sdot1RowPanelFn = @convention(c) (
        UnsafePointer<Int32>, UnsafePointer<Int8>, Int, UnsafeMutablePointer<Int32>
    ) -> Void

    @usableFromInline let gemmSmeRowblock: GemmSmeRowblockFn
    @usableFromInline let attentionQktSme: AttentionQktSmeFn
    @usableFromInline let attentionPvSme: AttentionPvSmeFn
    @usableFromInline let f32ToF16Neon: F32ToF16NeonFn
    @usableFromInline let geluF16Pass1: GeluF16Pass1Fn
    @usableFromInline let geluF16Pass2: GeluF16Pass2Fn
    @usableFromInline let f16x8ToF32x8: F16x8ToF32x8Fn
    @usableFromInline let smmla8x8Tile: SmmlaTileFn
    @usableFromInline let smmla4x8Tile: SmmlaTileFn
    @usableFromInline let smmla2x8Tile: SmmlaTileFn
    @usableFromInline let smmlaM12Panel: SmmlaM12PanelFn
    @usableFromInline let sdot4RowPanel: Sdot4RowPanelFn
    @usableFromInline let sdot1RowPanel: Sdot1RowPanelFn
    private let pagePtr: UnsafeMutableRawPointer
    private let pageSize: Int

    private init() {
        let ps = Int(getpagesize())
        let totalBytes = max(ps, 2048)
        self.pageSize = totalBytes
        let mem = mmap(nil, totalBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)!
        self.pagePtr = mem
        let dst = mem.assumingMemoryBound(to: UInt32.self)
        Self.writeCode(to: dst)
        let rc = mprotect(mem, totalBytes, PROT_READ | PROT_EXEC)
        precondition(rc == 0, "mprotect PROT_EXEC failed")
        #if canImport(Darwin)
        sys_icache_invalidate(mem, totalBytes)
        #elseif canImport(Glibc)
        clear_cache(mem, mem.advanced(by: totalBytes))
        #endif
        self.gemmSmeRowblock = unsafeBitCast(mem.advanced(by: 0), to: GemmSmeRowblockFn.self)
        self.attentionQktSme = unsafeBitCast(mem.advanced(by: 384), to: AttentionQktSmeFn.self)
        self.attentionPvSme = unsafeBitCast(mem.advanced(by: 624), to: AttentionPvSmeFn.self)
        self.f32ToF16Neon = unsafeBitCast(mem.advanced(by: 816), to: F32ToF16NeonFn.self)
        self.geluF16Pass1 = unsafeBitCast(mem.advanced(by: 864), to: GeluF16Pass1Fn.self)
        self.geluF16Pass2 = unsafeBitCast(mem.advanced(by: 1248), to: GeluF16Pass2Fn.self)
        self.f16x8ToF32x8 = unsafeBitCast(mem.advanced(by: 1312), to: F16x8ToF32x8Fn.self)
        self.smmla8x8Tile = unsafeBitCast(mem.advanced(by: 1344), to: SmmlaTileFn.self)
        self.smmla4x8Tile = unsafeBitCast(mem.advanced(by: 1584), to: SmmlaTileFn.self)
        self.smmla2x8Tile = unsafeBitCast(mem.advanced(by: 1696), to: SmmlaTileFn.self)
        self.smmlaM12Panel = unsafeBitCast(mem.advanced(by: 1760), to: SmmlaM12PanelFn.self)
        self.sdot4RowPanel = unsafeBitCast(mem.advanced(by: 1872), to: Sdot4RowPanelFn.self)
        self.sdot1RowPanel = unsafeBitCast(mem.advanced(by: 1984), to: Sdot1RowPanelFn.self)
    }

    deinit {
        munmap(pagePtr, pageSize)
    }

    private static func writeCode(to dst: UnsafeMutablePointer<UInt32>) {
        // gemm_sme_rowblock @ word offset 0 (byte offset 0)
        let p_gemm_sme_rowblock: [UInt32] = [
            0xd10103ff, // sub sp, sp, #0x40
            0x6d0027e8, // stp d8, d9, [sp]
            0x6d012fea, // stp d10, d11, [sp, #0x10]
            0x6d0237ec, // stp d12, d13, [sp, #0x20]
            0x6d033fee, // stp d14, d15, [sp, #0x30]
            0x92403ce8, // and x8, x7, #0xffff
            0xd350fce9, // lsr x9, x7, #16
            0x92403d29, // and x9, x9, #0xffff
            0xd360fcea, // lsr x10, x7, #32
            0x92403d4a, // and x10, x10, #0xffff
            0xd370fceb, // lsr x11, x7, #48
            0xd503477f, // smstart
            0x2518e3e0, // ptrue p0.b
            0x2598e3e1, // ptrue p1.s
            0xc00800ff, // zero {za}
            0xaa0803ed, // mov x13, x8
            0xaa0003ee, // mov x14, x0
            0xa400a1c0, // ld1b { z0.b }, p0/z, [x14]
            0xa400a021, // ld1b { z1.b }, p0/z, [x1]
            0xa401a022, // ld1b { z2.b }, p0/z, [x1, #0x1, mul vl]
            0xa402a023, // ld1b { z3.b }, p0/z, [x1, #0x2, mul vl]
            0xa403a024, // ld1b { z4.b }, p0/z, [x1, #0x3, mul vl]
            0x910101ce, // add x14, x14, #0x40
            0x91040021, // add x1, x1, #0x100
            0xa0810000, // smopa za0.s, p0/m, p0/m, z0.b, z1.b
            0xa0820001, // smopa za1.s, p0/m, p0/m, z0.b, z2.b
            0xa0830002, // smopa za2.s, p0/m, p0/m, z0.b, z3.b
            0xa0840003, // smopa za3.s, p0/m, p0/m, z0.b, z4.b
            0xf10005ad, // subs x13, x13, #0x1
            0x54fffe81, // b.ne 0x44
            0xa540a450, // ld1w { z16.s }, p1/z, [x2]
            0xa541a451, // ld1w { z17.s }, p1/z, [x2, #0x1, mul vl]
            0xa542a452, // ld1w { z18.s }, p1/z, [x2, #0x2, mul vl]
            0xa543a453, // ld1w { z19.s }, p1/z, [x2, #0x3, mul vl]
            0xa540a474, // ld1w { z20.s }, p1/z, [x3]
            0xa541a475, // ld1w { z21.s }, p1/z, [x3, #0x1, mul vl]
            0xa542a476, // ld1w { z22.s }, p1/z, [x3, #0x2, mul vl]
            0xa543a477, // ld1w { z23.s }, p1/z, [x3, #0x3, mul vl]
            0x91040042, // add x2, x2, #0x100
            0x91040063, // add x3, x3, #0x100
            0xaa0403ef, // mov x15, x4
            0xaa0603f0, // mov x16, x6
            0xaa0503f1, // mov x17, x5
            0x5280000c, // mov w12, #0x0
            0xc0820400, // mov z0.s, p1/m, za0h.s[w12, 0]
            0xc0820481, // mov z1.s, p1/m, za1h.s[w12, 0]
            0xc0820502, // mov z2.s, p1/m, za2h.s[w12, 0]
            0xc0820583, // mov z3.s, p1/m, za3h.s[w12, 0]
            0x6594a400, // scvtf z0.s, p1/m, z0.s
            0x6594a421, // scvtf z1.s, p1/m, z1.s
            0x6594a442, // scvtf z2.s, p1/m, z2.s
            0x6594a463, // scvtf z3.s, p1/m, z3.s
            0x8540c5e4, // ld1rw { z4.s }, p1/z, [x15]
            0x910011ef, // add x15, x15, #0x4
            0x65840a05, // fmul z5.s, z16.s, z4.s
            0x65840a26, // fmul z6.s, z17.s, z4.s
            0x65840a47, // fmul z7.s, z18.s, z4.s
            0x65840a68, // fmul z8.s, z19.s, z4.s
            0x04743298, // mov z24.d, z20.d
            0x047532b9, // mov z25.d, z21.d
            0x047632da, // mov z26.d, z22.d
            0x047732fb, // mov z27.d, z23.d
            0xb4000151, // cbz x17, 0x120
            0xa540a63c, // ld1w { z28.s }, p1/z, [x17]
            0xa541a63d, // ld1w { z29.s }, p1/z, [x17, #0x1, mul vl]
            0xa542a63e, // ld1w { z30.s }, p1/z, [x17, #0x2, mul vl]
            0xa543a63f, // ld1w { z31.s }, p1/z, [x17, #0x3, mul vl]
            0x65808798, // fadd z24.s, p1/m, z24.s, z28.s
            0x658087b9, // fadd z25.s, p1/m, z25.s, z29.s
            0x658087da, // fadd z26.s, p1/m, z26.s, z30.s
            0x658087fb, // fadd z27.s, p1/m, z27.s, z31.s
            0x8b090231, // add x17, x17, x9
            0x65a50418, // fmla z24.s, p1/m, z0.s, z5.s
            0x65a60439, // fmla z25.s, p1/m, z1.s, z6.s
            0x65a7045a, // fmla z26.s, p1/m, z2.s, z7.s
            0x65a8047b, // fmla z27.s, p1/m, z3.s, z8.s
            0xe540e618, // st1w { z24.s }, p1, [x16]
            0xe541e619, // st1w { z25.s }, p1, [x16, #0x1, mul vl]
            0xe542e61a, // st1w { z26.s }, p1, [x16, #0x2, mul vl]
            0xe543e61b, // st1w { z27.s }, p1, [x16, #0x3, mul vl]
            0x8b090210, // add x16, x16, x9
            0x1100058c, // add w12, w12, #0x1
            0x6b0a019f, // cmp w12, w10
            0x54fffb23, // b.lo 0xb0
            0x910400c6, // add x6, x6, #0x100
            0xb4000045, // cbz x5, 0x15c
            0x910400a5, // add x5, x5, #0x100
            0xf100056b, // subs x11, x11, #0x1
            0x54fff6c1, // b.ne 0x38
            0xd503467f, // smstop
            0x6d4027e8, // ldp d8, d9, [sp]
            0x6d412fea, // ldp d10, d11, [sp, #0x10]
            0x6d4237ec, // ldp d12, d13, [sp, #0x20]
            0x6d433fee, // ldp d14, d15, [sp, #0x30]
            0x910103ff, // add sp, sp, #0x40
            0xd65f03c0, // ret
        ]
        for (i, w) in p_gemm_sme_rowblock.enumerated() { dst[0 + i] = w }
        // attention_qkt_sme @ word offset 96 (byte offset 384)
        let p_attention_qkt_sme: [UInt32] = [
            0xd10103ff, // sub sp, sp, #0x40
            0x6d0027e8, // stp d8, d9, [sp]
            0x6d012fea, // stp d10, d11, [sp, #0x10]
            0x6d0237ec, // stp d12, d13, [sp, #0x20]
            0x6d033fee, // stp d14, d15, [sp, #0x30]
            0x91080026, // add x6, x1, #0x200
            0x91100027, // add x7, x1, #0x400
            0x91180028, // add x8, x1, #0x600
            0xd503477f, // smstart
            0x2518e3e0, // ptrue p0.b
            0x2598e3e1, // ptrue p1.s
            0xc00800ff, // zero {za}
            0xd2800109, // mov x9, #0x8
            0xaa0003ea, // mov x10, x0
            0xa400a140, // ld1b { z0.b }, p0/z, [x10]
            0xa400a021, // ld1b { z1.b }, p0/z, [x1]
            0xa400a0c2, // ld1b { z2.b }, p0/z, [x6]
            0xa400a0e3, // ld1b { z3.b }, p0/z, [x7]
            0xa400a104, // ld1b { z4.b }, p0/z, [x8]
            0x9101014a, // add x10, x10, #0x40
            0x91010021, // add x1, x1, #0x40
            0x910100c6, // add x6, x6, #0x40
            0x910100e7, // add x7, x7, #0x40
            0x91010108, // add x8, x8, #0x40
            0xa0810000, // smopa za0.s, p0/m, p0/m, z0.b, z1.b
            0xa0820001, // smopa za1.s, p0/m, p0/m, z0.b, z2.b
            0xa0830002, // smopa za2.s, p0/m, p0/m, z0.b, z3.b
            0xa0840003, // smopa za3.s, p0/m, p0/m, z0.b, z4.b
            0xf1000529, // subs x9, x9, #0x1
            0x54fffe21, // b.ne 0x1b8
            0xaa0203eb, // mov x11, x2
            0x5280000c, // mov w12, #0x0
            0xe0bf0560, // st1w {za0h.s[w12, 0]}, p1, [x11]
            0x9101016d, // add x13, x11, #0x40
            0xe0bf05a4, // st1w {za1h.s[w12, 0]}, p1, [x13]
            0x9102016d, // add x13, x11, #0x80
            0xe0bf05a8, // st1w {za2h.s[w12, 0]}, p1, [x13]
            0x9103016d, // add x13, x11, #0xc0
            0xe0bf05ac, // st1w {za3h.s[w12, 0]}, p1, [x13]
            0x8b03016b, // add x11, x11, x3
            0x1100058c, // add w12, w12, #0x1
            0x6b05019f, // cmp w12, w5
            0x54fffec3, // b.lo 0x200
            0xaa0803e1, // mov x1, x8
            0x91080026, // add x6, x1, #0x200
            0x910800c7, // add x7, x6, #0x200
            0x910800e8, // add x8, x7, #0x200
            0x91040042, // add x2, x2, #0x100
            0xf1000484, // subs x4, x4, #0x1
            0x54fffb41, // b.ne 0x1ac
            0xd503467f, // smstop
            0x6d4027e8, // ldp d8, d9, [sp]
            0x6d412fea, // ldp d10, d11, [sp, #0x10]
            0x6d4237ec, // ldp d12, d13, [sp, #0x20]
            0x6d433fee, // ldp d14, d15, [sp, #0x30]
            0x910103ff, // add sp, sp, #0x40
            0xd65f03c0, // ret
        ]
        for (i, w) in p_attention_qkt_sme.enumerated() { dst[96 + i] = w }
        // attention_pv_sme @ word offset 156 (byte offset 624)
        let p_attention_pv_sme: [UInt32] = [
            0xd10103ff, // sub sp, sp, #0x40
            0x6d0027e8, // stp d8, d9, [sp]
            0x6d012fea, // stp d10, d11, [sp, #0x10]
            0x6d0237ec, // stp d12, d13, [sp, #0x20]
            0x6d033fee, // stp d14, d15, [sp, #0x30]
            0xd503477f, // smstart
            0x2518e3e0, // ptrue p0.b
            0x2598e3e1, // ptrue p1.s
            0xc00800ff, // zero {za}
            0xaa0403e7, // mov x7, x4
            0xa400a000, // ld1b { z0.b }, p0/z, [x0]
            0xa400a021, // ld1b { z1.b }, p0/z, [x1]
            0xa401a022, // ld1b { z2.b }, p0/z, [x1, #0x1, mul vl]
            0x91010000, // add x0, x0, #0x40
            0x91020021, // add x1, x1, #0x80
            0xa0810008, // smopa za0.s, p0/m, p0/m, z0.h, z1.h
            0xa0820009, // smopa za1.s, p0/m, p0/m, z0.h, z2.h
            0xf10004e7, // subs x7, x7, #0x1
            0x54ffff01, // b.ne 0x28c
            0xa540a450, // ld1w { z16.s }, p1/z, [x2]
            0xa541a451, // ld1w { z17.s }, p1/z, [x2, #0x1, mul vl]
            0x5280000c, // mov w12, #0x0
            0xc0820400, // mov z0.s, p1/m, za0h.s[w12, 0]
            0xc0820481, // mov z1.s, p1/m, za1h.s[w12, 0]
            0x6594a400, // scvtf z0.s, p1/m, z0.s
            0x6594a421, // scvtf z1.s, p1/m, z1.s
            0x8540c464, // ld1rw { z4.s }, p1/z, [x3]
            0x91001063, // add x3, x3, #0x4
            0x65840a05, // fmul z5.s, z16.s, z4.s
            0x65840a26, // fmul z6.s, z17.s, z4.s
            0x65850800, // fmul z0.s, z0.s, z5.s
            0x65860821, // fmul z1.s, z1.s, z6.s
            0xe540e4a0, // st1w { z0.s }, p1, [x5]
            0xe541e4a1, // st1w { z1.s }, p1, [x5, #0x1, mul vl]
            0x911800a5, // add x5, x5, #0x600
            0x1100058c, // add w12, w12, #0x1
            0x6b06019f, // cmp w12, w6
            0x54fffe23, // b.lo 0x2bc
            0xd503467f, // smstop
            0x6d4027e8, // ldp d8, d9, [sp]
            0x6d412fea, // ldp d10, d11, [sp, #0x10]
            0x6d4237ec, // ldp d12, d13, [sp, #0x20]
            0x6d433fee, // ldp d14, d15, [sp, #0x30]
            0x910103ff, // add sp, sp, #0x40
            0xd65f03c0, // ret
        ]
        for (i, w) in p_attention_pv_sme.enumerated() { dst[156 + i] = w }
        // f32_to_f16_neon @ word offset 204 (byte offset 816)
        let p_f32_to_f16_neon: [UInt32] = [
            0x4cdf2800, // ld1.4s { v0, v1, v2, v3 }, [x0], #64
            0x0e216804, // fcvtn v4.4h, v0.4s
            0x4e216824, // fcvtn2 v4.8h, v1.4s
            0x0e216845, // fcvtn v5.4h, v2.4s
            0x4e216865, // fcvtn2 v5.8h, v3.4s
            0x4c9fa424, // st1.8h { v4, v5 }, [x1], #32
            0xf1000442, // subs x2, x2, #0x1
            0x54ffff21, // b.ne _f32_to_f16_neon
            0xd65f03c0, // ret
        ]
        for (i, w) in p_f32_to_f16_neon.enumerated() { dst[204 + i] = w }
        // gelu_f16_pass1 @ word offset 216 (byte offset 864)
        let p_gelu_f16_pass1: [UInt32] = [
            0xd10103ff, // sub sp, sp, #0x40
            0x6d0027e8, // stp d8, d9, [sp]
            0x6d012fea, // stp d10, d11, [sp, #0x10]
            0x6d0237ec, // stp d12, d13, [sp, #0x20]
            0x6d033fee, // stp d14, d15, [sp, #0x30]
            0xaa0103e4, // mov x4, x1
            0x4c406810, // ld1.4s { v16, v17, v18 }, [x0]
            0x0e216a1d, // fcvtn v29.4h, v16.4s
            0x4e216a1d, // fcvtn2 v29.8h, v16.4s
            0x0e216a3e, // fcvtn v30.4h, v17.4s
            0x4e216a3e, // fcvtn2 v30.8h, v17.4s
            0x0e216a5f, // fcvtn v31.4h, v18.4s
            0x4e216a5f, // fcvtn2 v31.8h, v18.4s
            0x4e0207b0, // dup.8h v16, v29[0]
            0x4e0607b1, // dup.8h v17, v29[1]
            0x4e0a07b2, // dup.8h v18, v29[2]
            0x4e0e07b3, // dup.8h v19, v29[3]
            0x4e0207d4, // dup.8h v20, v30[0]
            0x4e0607d5, // dup.8h v21, v30[1]
            0x4e0a07d6, // dup.8h v22, v30[2]
            0x4e0e07d7, // dup.8h v23, v30[3]
            0x4e0207f8, // dup.8h v24, v31[0]
            0x4e0607f9, // dup.8h v25, v31[1]
            0x4e0a07fa, // dup.8h v26, v31[2]
            0x4f00841b, // movi.8h v27, #0x0
            0x4f00841c, // movi.8h v28, #0x0
            0x4cdf2820, // ld1.4s { v0, v1, v2, v3 }, [x1], #64
            0x4cdfa446, // ld1.8h { v6, v7 }, [x2], #32
            0x0e216804, // fcvtn v4.4h, v0.4s
            0x4e216824, // fcvtn2 v4.8h, v1.4s
            0x0e216845, // fcvtn v5.4h, v2.4s
            0x4e216865, // fcvtn2 v5.8h, v3.4s
            0x4e503480, // fmax.8h v0, v4, v16
            0x4e5034a1, // fmax.8h v1, v5, v16
            0x4ed13400, // fmin.8h v0, v0, v17
            0x4ed13421, // fmin.8h v1, v1, v17
            0x6e401c02, // fmul.8h v2, v0, v0
            0x6e411c23, // fmul.8h v3, v1, v1
            0x4eb31e68, // mov.16b v8, v19
            0x4eb31e69, // mov.16b v9, v19
            0x4e520c48, // fmla.8h v8, v2, v18
            0x4e520c69, // fmla.8h v9, v3, v18
            0x4eb41e8a, // mov.16b v10, v20
            0x4eb41e8b, // mov.16b v11, v20
            0x4e480c4a, // fmla.8h v10, v2, v8
            0x4e490c6b, // fmla.8h v11, v3, v9
            0x4eb51ea8, // mov.16b v8, v21
            0x4eb51ea9, // mov.16b v9, v21
            0x4e4a0c48, // fmla.8h v8, v2, v10
            0x4e4b0c69, // fmla.8h v9, v3, v11
            0x4eb71eea, // mov.16b v10, v23
            0x4eb71eeb, // mov.16b v11, v23
            0x4e560c4a, // fmla.8h v10, v2, v22
            0x4e560c6b, // fmla.8h v11, v3, v22
            0x4eb81f0c, // mov.16b v12, v24
            0x4eb81f0d, // mov.16b v13, v24
            0x4e4a0c4c, // fmla.8h v12, v2, v10
            0x4e4b0c6d, // fmla.8h v13, v3, v11
            0x4eb91f2a, // mov.16b v10, v25
            0x4eb91f2b, // mov.16b v11, v25
            0x4e4c0c4a, // fmla.8h v10, v2, v12
            0x4e4d0c6b, // fmla.8h v11, v3, v13
            0x4ef9d94c, // frecpe.8h v12, v10
            0x4ef9d96d, // frecpe.8h v13, v11
            0x4e4c3d4a, // frecps.8h v10, v10, v12
            0x4e4d3d6b, // frecps.8h v11, v11, v13
            0x6e4a1d8c, // fmul.8h v12, v12, v10
            0x6e4b1dad, // fmul.8h v13, v13, v11
            0x6e4c1d08, // fmul.8h v8, v8, v12
            0x6e4d1d29, // fmul.8h v9, v9, v13
            0x4eba1f4a, // mov.16b v10, v26
            0x4eba1f4b, // mov.16b v11, v26
            0x4e480c0a, // fmla.8h v10, v0, v8
            0x4e490c2b, // fmla.8h v11, v1, v9
            0x6e4a1c84, // fmul.8h v4, v4, v10
            0x6e4b1ca5, // fmul.8h v5, v5, v11
            0x6e461c84, // fmul.8h v4, v4, v6
            0x6e471ca5, // fmul.8h v5, v5, v7
            0x4c9fa484, // st1.8h { v4, v5 }, [x4], #32
            0x4ef8f884, // fabs.8h v4, v4
            0x4ef8f8a5, // fabs.8h v5, v5
            0x4e44377b, // fmax.8h v27, v27, v4
            0x4e45379c, // fmax.8h v28, v28, v5
            0xf1000463, // subs x3, x3, #0x1
            0x54fff8c1, // b.ne 0x3a4
            0x4e5c377b, // fmax.8h v27, v27, v28
            0x4e30fb7b, // fmaxv.8h h27, v27
            0x1ee2437b, // fcvt s27, h27
            0x1e260360, // fmov w0, s27
            0x6d4027e8, // ldp d8, d9, [sp]
            0x6d412fea, // ldp d10, d11, [sp, #0x10]
            0x6d4237ec, // ldp d12, d13, [sp, #0x20]
            0x6d433fee, // ldp d14, d15, [sp, #0x30]
            0x910103ff, // add sp, sp, #0x40
            0xd65f03c0, // ret
        ]
        for (i, w) in p_gelu_f16_pass1.enumerated() { dst[216 + i] = w }
        // gelu_f16_pass2 @ word offset 312 (byte offset 1248)
        let p_gelu_f16_pass2: [UInt32] = [
            0x1e27001d, // fmov s29, w0
            0x1e23c3bd, // fcvt h29, s29
            0x4e0207bd, // dup.8h v29, v29[0]
            0x4cdfa420, // ld1.8h { v0, v1 }, [x1], #32
            0x6e5d1c00, // fmul.8h v0, v0, v29
            0x6e5d1c21, // fmul.8h v1, v1, v29
            0x4e79a800, // fcvtns.8h v0, v0
            0x4e79a821, // fcvtns.8h v1, v1
            0x0e214802, // sqxtn.8b v2, v0
            0x4e214822, // sqxtn2.16b v2, v1
            0x4c9f7042, // st1.16b { v2 }, [x2], #16
            0xf1000463, // subs x3, x3, #0x1
            0x54fffee1, // b.ne 0x4c4
            0xd65f03c0, // ret
        ]
        for (i, w) in p_gelu_f16_pass2.enumerated() { dst[312 + i] = w }
        // f16x8_to_f32x8 @ word offset 328 (byte offset 1312)
        let p_f16x8_to_f32x8: [UInt32] = [
            0x4c407400, // ld1.8h { v0 }, [x0]
            0x0e217801, // fcvtl v1.4s, v0.4h
            0x4e217802, // fcvtl2 v2.4s, v0.8h
            0x4c00a821, // st1.4s { v1, v2 }, [x1]
            0xd65f03c0, // ret
        ]
        for (i, w) in p_f16x8_to_f32x8.enumerated() { dst[328 + i] = w }
        // smmla_8x8_tile @ word offset 336 (byte offset 1344)
        let p_smmla_8x8_tile: [UInt32] = [
            0xd10103ff, // sub sp, sp, #0x40
            0x6d0027e8, // stp d8, d9, [sp]
            0x6d012fea, // stp d10, d11, [sp, #0x10]
            0x6d0237ec, // stp d12, d13, [sp, #0x20]
            0x6d033fee, // stp d14, d15, [sp, #0x30]
            0x6f00e400, // movi.2d v0, #0000000000000000
            0x6f00e401, // movi.2d v1, #0000000000000000
            0x6f00e402, // movi.2d v2, #0000000000000000
            0x6f00e403, // movi.2d v3, #0000000000000000
            0x6f00e404, // movi.2d v4, #0000000000000000
            0x6f00e405, // movi.2d v5, #0000000000000000
            0x6f00e406, // movi.2d v6, #0000000000000000
            0x6f00e407, // movi.2d v7, #0000000000000000
            0x6f00e408, // movi.2d v8, #0000000000000000
            0x6f00e409, // movi.2d v9, #0000000000000000
            0x6f00e40a, // movi.2d v10, #0000000000000000
            0x6f00e40b, // movi.2d v11, #0000000000000000
            0x6f00e40c, // movi.2d v12, #0000000000000000
            0x6f00e40d, // movi.2d v13, #0000000000000000
            0x6f00e40e, // movi.2d v14, #0000000000000000
            0x6f00e40f, // movi.2d v15, #0000000000000000
            0xacc14410, // ldp q16, q17, [x0], #0x20
            0xacc15434, // ldp q20, q21, [x1], #0x20
            0xacc14c12, // ldp q18, q19, [x0], #0x20
            0x4e94a600, // smmla
            0x4e95a601, // smmla
            0xacc15c36, // ldp q22, q23, [x1], #0x20
            0x4e94a624, // smmla
            0x4e95a625, // smmla
            0x4e96a602, // smmla
            0x4e97a603, // smmla
            0x4e96a626, // smmla
            0x4e97a627, // smmla
            0x4e94a648, // smmla
            0x4e95a649, // smmla
            0x4e96a64a, // smmla
            0x4e97a64b, // smmla
            0x4e94a66c, // smmla
            0x4e95a66d, // smmla
            0x4e96a66e, // smmla
            0x4e97a66f, // smmla
            0xf1000442, // subs x2, x2, #0x1
            0x54fffd61, // b.ne 0x558
            0xad000460, // stp q0, q1, [x3]
            0xad010c62, // stp q2, q3, [x3, #0x20]
            0xad021464, // stp q4, q5, [x3, #0x40]
            0xad031c66, // stp q6, q7, [x3, #0x60]
            0xad042468, // stp q8, q9, [x3, #0x80]
            0xad052c6a, // stp q10, q11, [x3, #0xa0]
            0xad06346c, // stp q12, q13, [x3, #0xc0]
            0xad073c6e, // stp q14, q15, [x3, #0xe0]
            0x6d4027e8, // ldp d8, d9, [sp]
            0x6d412fea, // ldp d10, d11, [sp, #0x10]
            0x6d4237ec, // ldp d12, d13, [sp, #0x20]
            0x6d433fee, // ldp d14, d15, [sp, #0x30]
            0x910103ff, // add sp, sp, #0x40
            0xd65f03c0, // ret
        ]
        for (i, w) in p_smmla_8x8_tile.enumerated() { dst[336 + i] = w }
        // smmla_4x8_tile @ word offset 396 (byte offset 1584)
        let p_smmla_4x8_tile: [UInt32] = [
            0x6f00e400, // movi.2d v0, #0000000000000000
            0x6f00e401, // movi.2d v1, #0000000000000000
            0x6f00e402, // movi.2d v2, #0000000000000000
            0x6f00e403, // movi.2d v3, #0000000000000000
            0x6f00e404, // movi.2d v4, #0000000000000000
            0x6f00e405, // movi.2d v5, #0000000000000000
            0x6f00e406, // movi.2d v6, #0000000000000000
            0x6f00e407, // movi.2d v7, #0000000000000000
            0xacc14410, // ldp q16, q17, [x0], #0x20
            0xacc15434, // ldp q20, q21, [x1], #0x20
            0xacc15c36, // ldp q22, q23, [x1], #0x20
            0x4e94a600, // smmla
            0x4e95a601, // smmla
            0x4e96a602, // smmla
            0x4e97a603, // smmla
            0x4e94a624, // smmla
            0x4e95a625, // smmla
            0x4e96a626, // smmla
            0x4e97a627, // smmla
            0xf1000442, // subs x2, x2, #0x1
            0x54fffe81, // b.ne 0x608
            0xad000460, // stp q0, q1, [x3]
            0xad010c62, // stp q2, q3, [x3, #0x20]
            0xad021464, // stp q4, q5, [x3, #0x40]
            0xad031c66, // stp q6, q7, [x3, #0x60]
            0xd65f03c0, // ret
        ]
        for (i, w) in p_smmla_4x8_tile.enumerated() { dst[396 + i] = w }
        // smmla_2x8_tile @ word offset 424 (byte offset 1696)
        let p_smmla_2x8_tile: [UInt32] = [
            0x6f00e400, // movi.2d v0, #0000000000000000
            0x6f00e401, // movi.2d v1, #0000000000000000
            0x6f00e402, // movi.2d v2, #0000000000000000
            0x6f00e403, // movi.2d v3, #0000000000000000
            0x3cc10410, // ldr q16, [x0], #0x10
            0xacc15434, // ldp q20, q21, [x1], #0x20
            0xacc15c36, // ldp q22, q23, [x1], #0x20
            0x4e94a600, // smmla
            0x4e95a601, // smmla
            0x4e96a602, // smmla
            0x4e97a603, // smmla
            0xf1000442, // subs x2, x2, #0x1
            0x54ffff01, // b.ne 0x660
            0xad000460, // stp q0, q1, [x3]
            0xad010c62, // stp q2, q3, [x3, #0x20]
            0xd65f03c0, // ret
        ]
        for (i, w) in p_smmla_2x8_tile.enumerated() { dst[424 + i] = w }
        // smmla_m1_2panel @ word offset 440 (byte offset 1760)
        let p_smmla_m1_2panel: [UInt32] = [
            0x6f00e400, // movi.2d v0, #0000000000000000
            0x6f00e401, // movi.2d v1, #0000000000000000
            0x6f00e402, // movi.2d v2, #0000000000000000
            0x6f00e403, // movi.2d v3, #0000000000000000
            0x6f00e404, // movi.2d v4, #0000000000000000
            0x6f00e405, // movi.2d v5, #0000000000000000
            0x6f00e406, // movi.2d v6, #0000000000000000
            0x6f00e407, // movi.2d v7, #0000000000000000
            0x3cc10410, // ldr q16, [x0], #0x10
            0xacc15434, // ldp q20, q21, [x1], #0x20
            0xacc15c36, // ldp q22, q23, [x1], #0x20
            0xacc16458, // ldp q24, q25, [x2], #0x20
            0xacc16c5a, // ldp q26, q27, [x2], #0x20
            0x4e94a600, // smmla
            0x4e95a601, // smmla
            0x4e96a602, // smmla
            0x4e97a603, // smmla
            0x4e98a604, // smmla
            0x4e99a605, // smmla
            0x4e9aa606, // smmla
            0x4e9ba607, // smmla
            0xf1000463, // subs x3, x3, #0x1
            0x54fffe41, // b.ne 0x6b0
            0xad000480, // stp q0, q1, [x4]
            0xad010c82, // stp q2, q3, [x4, #0x20]
            0xad021484, // stp q4, q5, [x4, #0x40]
            0xad031c86, // stp q6, q7, [x4, #0x60]
            0xd65f03c0, // ret
        ]
        for (i, w) in p_smmla_m1_2panel.enumerated() { dst[440 + i] = w }
        // sdot_4row_panel @ word offset 468 (byte offset 1872)
        let p_sdot_4row_panel: [UInt32] = [
            0x6f00e400, // movi.2d v0, #0000000000000000
            0x6f00e401, // movi.2d v1, #0000000000000000
            0x6f00e402, // movi.2d v2, #0000000000000000
            0x6f00e403, // movi.2d v3, #0000000000000000
            0x6f00e404, // movi.2d v4, #0000000000000000
            0x6f00e405, // movi.2d v5, #0000000000000000
            0x6f00e406, // movi.2d v6, #0000000000000000
            0x6f00e407, // movi.2d v7, #0000000000000000
            0xacc14490, // ldp q16, q17, [x4], #0x20
            0x4ddfc814, // ld1r.4s { v20 }, [x0], #4
            0x4ddfc835, // ld1r.4s { v21 }, [x1], #4
            0x4ddfc856, // ld1r.4s { v22 }, [x2], #4
            0x4ddfc877, // ld1r.4s { v23 }, [x3], #4
            0x4e949600, // sdot.4s v0, v16, v20
            0x4e949621, // sdot.4s v1, v17, v20
            0x4e959602, // sdot.4s v2, v16, v21
            0x4e959623, // sdot.4s v3, v17, v21
            0x4e969604, // sdot.4s v4, v16, v22
            0x4e969625, // sdot.4s v5, v17, v22
            0x4e979606, // sdot.4s v6, v16, v23
            0x4e979627, // sdot.4s v7, v17, v23
            0xf10004a5, // subs x5, x5, #0x1
            0x54fffe41, // b.ne 0x720
            0xad0004c0, // stp q0, q1, [x6]
            0xad010cc2, // stp q2, q3, [x6, #0x20]
            0xad0214c4, // stp q4, q5, [x6, #0x40]
            0xad031cc6, // stp q6, q7, [x6, #0x60]
            0xd65f03c0, // ret
        ]
        for (i, w) in p_sdot_4row_panel.enumerated() { dst[468 + i] = w }
        // sdot_1row_panel @ word offset 496 (byte offset 1984)
        let p_sdot_1row_panel: [UInt32] = [
            0x6f00e400, // movi.2d v0, #0000000000000000
            0x6f00e401, // movi.2d v1, #0000000000000000
            0xacc14430, // ldp q16, q17, [x1], #0x20
            0x4ddfc814, // ld1r.4s { v20 }, [x0], #4
            0x4e949600, // sdot.4s v0, v16, v20
            0x4e949621, // sdot.4s v1, v17, v20
            0xf1000442, // subs x2, x2, #0x1
            0x54ffff61, // b.ne 0x778
            0xad000460, // stp q0, q1, [x3]
            0xd65f03c0, // ret
        ]
        for (i, w) in p_sdot_1row_panel.enumerated() { dst[496 + i] = w }
    }
}
#endif
