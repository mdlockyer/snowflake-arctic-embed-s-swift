import Foundation

/// 64-byte cache-line aligned, owned buffer for SIMD kernels (ARM64 SME/I8MM/NEON & x86_64 AVX2/VNNI).
public final class AlignedBuf<Element>: @unchecked Sendable {
    public static var align: Int { 64 }

    public let ptr: UnsafeMutablePointer<Element>
    public let count: Int
    @usableFromInline let rawPtr: UnsafeMutableRawPointer?
    @usableFromInline let byteCapacity: Int

    @inlinable
    public var len: Int { count }

    @inlinable
    public var isEmpty: Bool { count == 0 }

    @inlinable
    public init(zeroed count: Int) {
        self.count = count
        if count <= 0 {
            self.rawPtr = nil
            self.byteCapacity = 0
            self.ptr = UnsafeMutablePointer<Element>(bitPattern: Self.align)!
            return
        }
        let elemStride = max(MemoryLayout<Element>.stride, 1)
        let alignment = max(Self.align, MemoryLayout<Element>.alignment)
        let rawBytes = count * elemStride
        let alignedBytes = (rawBytes + alignment - 1) & ~(alignment - 1)
        let raw = UnsafeMutableRawPointer.allocate(byteCount: alignedBytes, alignment: alignment)
        raw.initializeMemory(as: UInt8.self, repeating: 0, count: alignedBytes)
        self.rawPtr = raw
        self.byteCapacity = alignedBytes
        self.ptr = raw.assumingMemoryBound(to: Element.self)
    }

    @inlinable
    public static func newZeroed(_ len: Int) -> AlignedBuf<Element> {
        AlignedBuf<Element>(zeroed: len)
    }

    @inlinable
    public static func fromSlice(_ src: [Element]) -> AlignedBuf<Element> {
        if src.isEmpty {
            return AlignedBuf<Element>(zeroed: 0)
        }
        let buf = AlignedBuf<Element>(zeroed: src.count)
        src.withUnsafeBufferPointer { srcBuf in
            if let base = srcBuf.baseAddress {
                UnsafeMutableRawPointer(buf.ptr).copyMemory(
                    from: UnsafeRawPointer(base),
                    byteCount: src.count * MemoryLayout<Element>.stride
                )
            }
        }
        return buf
    }

    @inlinable
    public static func fromSlice(_ src: ArraySlice<Element>) -> AlignedBuf<Element> {
        if src.isEmpty {
            return AlignedBuf<Element>(zeroed: 0)
        }
        let buf = AlignedBuf<Element>(zeroed: src.count)
        src.withUnsafeBufferPointer { srcBuf in
            if let base = srcBuf.baseAddress {
                UnsafeMutableRawPointer(buf.ptr).copyMemory(
                    from: UnsafeRawPointer(base),
                    byteCount: src.count * MemoryLayout<Element>.stride
                )
            }
        }
        return buf
    }

    @inlinable
    public func clone() -> AlignedBuf<Element> {
        if count == 0 {
            return AlignedBuf<Element>(zeroed: 0)
        }
        let copy = AlignedBuf<Element>(zeroed: count)
        UnsafeMutableRawPointer(copy.ptr).copyMemory(
            from: UnsafeRawPointer(ptr),
            byteCount: count * MemoryLayout<Element>.stride
        )
        return copy
    }

    @inlinable
    public func asPtr() -> UnsafePointer<Element> {
        UnsafePointer(ptr)
    }

    @inlinable
    public func asMutPtr() -> UnsafeMutablePointer<Element> {
        ptr
    }

    @inlinable
    public subscript(index: Int) -> Element {
        get {
            ptr[index]
        }
        set {
            ptr[index] = newValue
        }
    }

    @inlinable
    public func copyWithin(srcRange: Range<Int>, destStart: Int) {
        let n = srcRange.count
        if n <= 0 || srcRange.lowerBound == destStart { return }
        let stride = MemoryLayout<Element>.stride
        memmove(
            UnsafeMutableRawPointer(ptr.advanced(by: destStart)),
            UnsafeRawPointer(ptr.advanced(by: srcRange.lowerBound)),
            n * stride
        )
    }

    deinit {
        if let rawPtr = rawPtr {
            rawPtr.deallocate()
        }
    }
}
