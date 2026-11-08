// Auto-assembled x86_64 CPUID + AVX2 / AVX-VNNI machine-code kernels.
#if arch(x86_64)
import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

@usableFromInline
final class X86AsmKernels: @unchecked Sendable {
    @usableFromInline static let shared = X86AsmKernels()

    @usableFromInline typealias DetectFeaturesFn = @convention(c) () -> UInt32
    @usableFromInline typealias Avx2PanelFn = @convention(c) (
        UnsafePointer<Int32>, UnsafePointer<Int8>, Int, UnsafeMutablePointer<Int32>
    ) -> Void
    @usableFromInline typealias AvxVnniPanelFn = @convention(c) (
        UnsafePointer<Int32>, UnsafePointer<Int8>, UnsafePointer<Int32>, Int, UnsafeMutablePointer<Int32>
    ) -> Void

    @usableFromInline let detectFeatures: DetectFeaturesFn
    @usableFromInline let avx2Panel: Avx2PanelFn
    @usableFromInline let avxVnniPanel: AvxVnniPanelFn
    @usableFromInline let featureMask: UInt32
    private let pagePtr: UnsafeMutableRawPointer
    private let pageSize: Int

    private init() {
        let ps = 4096
        self.pageSize = ps
        let mem = mmap(nil, ps, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)!
        self.pagePtr = mem
        let dst = mem.assumingMemoryBound(to: UInt8.self)
        Self.writeCode(to: dst)
        let rc = mprotect(mem, ps, PROT_READ | PROT_EXEC)
        precondition(rc == 0, "mprotect PROT_EXEC failed")
        let detect = unsafeBitCast(mem.advanced(by: 0), to: DetectFeaturesFn.self)
        self.detectFeatures = detect
        self.avx2Panel = unsafeBitCast(mem.advanced(by: 32), to: Avx2PanelFn.self)
        self.avxVnniPanel = unsafeBitCast(mem.advanced(by: 64), to: AvxVnniPanelFn.self)
        self.featureMask = detect()
    }

    deinit {
        munmap(pagePtr, pageSize)
    }

    private static func writeCode(to dst: UnsafeMutablePointer<UInt8>) {
        // x86_detect_features @ byte offset 0
        let p_x86_detect_features: [UInt8] = [
            0x53, // pushq %rbx
            0x31f6, // xorl %esi, %esi
            0xb801000000, // movl $0x1, %eax
            0x31c9, // xorl %ecx, %ecx
            0xfa2, // cpuid
            0xf7d1, // notl %ecx
            0xf7c100100018, // testl $0x18001000, %ecx               ## imm = 0x18001000
            0x7533, // jne 0x49
            0x31f6, // xorl %esi, %esi
            0x31c9, // xorl %ecx, %ecx
            0xf01d0, // xgetbv
            0xf7d0, // notl %eax
            0xa806, // testb $0x6, %al
            0x7526, // jne 0x49
            0x31f6, // xorl %esi, %esi
            0xb807000000, // movl $0x7, %eax
            0x31c9, // xorl %ecx, %ecx
            0xfa2, // cpuid
            0xf6c320, // testb $0x20, %bl
            0x7416, // je 0x49
            0xb807000000, // movl $0x7, %eax
            0xb901000000, // movl $0x1, %ecx
            0xfa2, // cpuid
            0x89c6, // movl %eax, %esi
            0x83e610, // andl $0x10, %esi
            0xc1ee03, // shrl $0x3, %esi
            0xffc6, // incl %esi
            0x89f0, // movl %esi, %eax
            0x5b, // popq %rbx
            0xc3, // retq
            0xf1f00, // nopl (%rax)
        ]
        for (i, b) in p_x86_detect_features.enumerated() { dst[0 + i] = b }
        // avx2_1row_panel @ byte offset 32
        let p_avx2_1row_panel: [UInt8] = [
            0x4885d2, // testq %rdx, %rdx
            0x7e5b, // jle 0xb0
            0xc5f9efc0, // vpxor %xmm0, %xmm0, %xmm0
            0x31c0, // xorl %eax, %eax
            0xc4e27d7915dc000000, // vpbroadcastw 0xdc(%rip), %ymm2
            0xc5f1efc9, // vpxor %xmm1, %xmm1, %xmm1
            0xf1f840000000000, // nopl (%rax,%rax)
            0xc4e27d581c87, // vpbroadcastd (%rdi,%rax,4), %ymm3
            0xc4e27d1ce3, // vpabsb %ymm3, %ymm4
            0xc5fe6f2e, // vmovdqu (%rsi), %ymm5
            0xc5fe6f7620, // vmovdqu 0x20(%rsi), %ymm6
            0xc4e25508eb, // vpsignb %ymm3, %ymm5, %ymm5
            0xc4e25d04ed, // vpmaddubsw %ymm5, %ymm4, %ymm5
            0xc4e25552c2, // {vex} vpdpwssd %ymm2, %ymm5, %ymm0
            0xc4e24d08db, // vpsignb %ymm3, %ymm6, %ymm3
            0xc4e25d04db, // vpmaddubsw %ymm3, %ymm4, %ymm3
            0xc4e26552ca, // {vex} vpdpwssd %ymm2, %ymm3, %ymm1
            0x48ffc0, // incq %rax
            0x4883c640, // addq $0x40, %rsi
            0x4839c2, // cmpq %rax, %rdx
            0x75c2, // jne 0x70
            0xeb08, // jmp 0xb8
            0xc5f1efc9, // vpxor %xmm1, %xmm1, %xmm1
            0xc5f9efc0, // vpxor %xmm0, %xmm0, %xmm0
            0xc5fe7f01, // vmovdqu %ymm0, (%rcx)
            0xc5fe7f4920, // vmovdqu %ymm1, 0x20(%rcx)
            0xc5f877, // vzeroupper
            0xc3, // retq
            0x66662e0f1f840000000000, // nopw %cs:(%rax,%rax)
        ]
        for (i, b) in p_avx2_1row_panel.enumerated() { dst[32 + i] = b }
        // avx_vnni_1row_panel @ byte offset 64
        let p_avx_vnni_1row_panel: [UInt8] = [
            0xc5fe6f0a, // vmovdqu (%rdx), %ymm1
            0xc5fe6f4220, // vmovdqu 0x20(%rdx), %ymm0
            0x4885c9, // testq %rcx, %rcx
            0x7e1f, // jle 0xfd
            0x31c0, // xorl %eax, %eax
            0xc4e27d581487, // vpbroadcastd (%rdi,%rax,4), %ymm2
            0xc4e26d500e, // {vex} vpdpbusd (%rsi), %ymm2, %ymm1
            0xc4e26d504620, // {vex} vpdpbusd 0x20(%rsi), %ymm2, %ymm0
            0x4883c640, // addq $0x40, %rsi
            0x48ffc0, // incq %rax
            0x4839c1, // cmpq %rax, %rcx
            0x75e3, // jne 0xe0
            0xc4c17e7f08, // vmovdqu %ymm1, (%r8)
            0xc4c17e7f4020, // vmovdqu %ymm0, 0x20(%r8)
            0xc5f877, // vzeroupper
            0xc3, // retq
        ]
        for (i, b) in p_avx_vnni_1row_panel.enumerated() { dst[64 + i] = b }
    }
}
#endif
