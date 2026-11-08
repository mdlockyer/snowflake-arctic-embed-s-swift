import Foundation

public let HIDDEN_SIZE: Int = 384
public let INTERMEDIATE_SIZE: Int = 1536
public let NUM_HEADS: Int = 12
public let HEAD_DIM: Int = 32
public let NUM_LAYERS: Int = 12
public let VOCAB_SIZE: Int = 30522
public let MAX_POSITIONS: Int = 512
public let TYPE_VOCAB_SIZE: Int = 2
public let LAYER_NORM_EPS: Float = 1e-12
public let SMOOTHQUANT_ALPHA: Float = 0.40

private let AE8_MAGIC: [UInt8] = [0x41, 0x52, 0x43, 0x54, 0x49, 0x43, 0x38, 0x01] // "ARCTIC8\x01"

public enum ArcticEmbedError: Error, CustomStringConvertible {
    case invalidMagic
    case unexpectedEOF
    case unsupportedBackend(String)
    case ioError(String)

    public var description: String {
        switch self {
        case .invalidMagic:
            return "invalid .ae8 magic header"
        case .unexpectedEOF:
            return "unexpected end of file while reading model data"
        case .unsupportedBackend(let name):
            return "SIMD backend \(name) is not supported by this CPU"
        case .ioError(let msg):
            return msg
        }
    }
}

public struct QuantizedLinearRaw: Sendable {
    public var nOut: Int
    public var kIn: Int
    public var wQ: [Int8]
    public var wScale: [Float]
    public var bias: [Float]
    public var invSmooth: [Float]

    public init(
        nOut: Int,
        kIn: Int,
        wQ: [Int8],
        wScale: [Float],
        bias: [Float],
        invSmooth: [Float]
    ) {
        self.nOut = nOut
        self.kIn = kIn
        self.wQ = wQ
        self.wScale = wScale
        self.bias = bias
        self.invSmooth = invSmooth
    }

    public func pack(_ backend: SimdBackend) -> PackedLinearInt8 {
        PackedLinearInt8.fromRawRowMajor(
            nOut: nOut,
            kIn: kIn,
            wRowMajor: wQ,
            wScale: wScale,
            bias: bias,
            invSmooth: invSmooth,
            backend: backend
        )
    }
}

public struct QuantizedLayerRaw: Sendable {
    public var qkv: QuantizedLinearRaw
    /// Separate Q (`[384, 384]`) and KV (`[768, 384]`) for Layer 11 CLS-only pruning.
    public var qOnly: QuantizedLinearRaw?
    public var kvOnly: QuantizedLinearRaw?
    public var attnOut: QuantizedLinearRaw
    public var ln1Weight: [Float]
    public var ln1Bias: [Float]
    public var ffnInter: QuantizedLinearRaw
    public var ffnOut: QuantizedLinearRaw
    public var ln2Weight: [Float]
    public var ln2Bias: [Float]
}

public struct QuantizedModelRaw: Sendable {
    public var wordEmbQ: [Int8]
    public var wordEmbScale: [Float]
    public var posEmbQ: [Int8]
    public var posEmbScale: [Float]
    public var typeEmbQ: [Int8]
    public var typeEmbScale: [Float]
    public var embLnWeight: [Float]
    public var embLnBias: [Float]
    public var layers: [QuantizedLayerRaw]

    /// Loads the quantized model from portable `.ae8` binary format.
    public static func loadAe8(_ path: String) throws -> QuantizedModelRaw {
        try loadAe8(URL(fileURLWithPath: path))
    }

    public static func loadAe8(_ url: URL) throws -> QuantizedModelRaw {
        let data = try Data(contentsOf: url)
        return try data.withUnsafeBytes { rawBuf in
            let bytes = rawBuf.bindMemory(to: UInt8.self)
            guard let base = bytes.baseAddress, bytes.count >= 8 else {
                throw ArcticEmbedError.invalidMagic
            }
            for i in 0..<8 {
                if base[i] != AE8_MAGIC[i] {
                    throw ArcticEmbedError.invalidMagic
                }
            }
            var cursor = 8
            let total = bytes.count

            let wordEmbQ = try readI8Vec(base, cursor: &cursor, total: total)
            let wordEmbScale = try readF32Vec(base, cursor: &cursor, total: total)
            let posEmbQ = try readI8Vec(base, cursor: &cursor, total: total)
            let posEmbScale = try readF32Vec(base, cursor: &cursor, total: total)
            let typeEmbQ = try readI8Vec(base, cursor: &cursor, total: total)
            let typeEmbScale = try readF32Vec(base, cursor: &cursor, total: total)
            let embLnWeight = try readF32Vec(base, cursor: &cursor, total: total)
            let embLnBias = try readF32Vec(base, cursor: &cursor, total: total)

            var layers: [QuantizedLayerRaw] = []
            layers.reserveCapacity(NUM_LAYERS)

            for _ in 0..<NUM_LAYERS {
                let qkv = try readLinear(base, cursor: &cursor, total: total)
                guard cursor < total else { throw ArcticEmbedError.unexpectedEOF }
                let hasSplit = base[cursor]
                cursor += 1
                let qOnly: QuantizedLinearRaw?
                let kvOnly: QuantizedLinearRaw?
                if hasSplit == 1 {
                    qOnly = try readLinear(base, cursor: &cursor, total: total)
                    kvOnly = try readLinear(base, cursor: &cursor, total: total)
                } else {
                    qOnly = nil
                    kvOnly = nil
                }
                let attnOut = try readLinear(base, cursor: &cursor, total: total)
                let ln1Weight = try readF32Vec(base, cursor: &cursor, total: total)
                let ln1Bias = try readF32Vec(base, cursor: &cursor, total: total)
                let ffnInter = try readLinear(base, cursor: &cursor, total: total)
                let ffnOut = try readLinear(base, cursor: &cursor, total: total)
                let ln2Weight = try readF32Vec(base, cursor: &cursor, total: total)
                let ln2Bias = try readF32Vec(base, cursor: &cursor, total: total)

                layers.append(
                    QuantizedLayerRaw(
                        qkv: qkv,
                        qOnly: qOnly,
                        kvOnly: kvOnly,
                        attnOut: attnOut,
                        ln1Weight: ln1Weight,
                        ln1Bias: ln1Bias,
                        ffnInter: ffnInter,
                        ffnOut: ffnOut,
                        ln2Weight: ln2Weight,
                        ln2Bias: ln2Bias
                    )
                )
            }

            return QuantizedModelRaw(
                wordEmbQ: wordEmbQ,
                wordEmbScale: wordEmbScale,
                posEmbQ: posEmbQ,
                posEmbScale: posEmbScale,
                typeEmbQ: typeEmbQ,
                typeEmbScale: typeEmbScale,
                embLnWeight: embLnWeight,
                embLnBias: embLnBias,
                layers: layers
            )
        }
    }

    /// Total memory footprint of quantized weights in bytes.
    public func sizeBytes() -> Int {
        var total = wordEmbQ.count
            + posEmbQ.count
            + typeEmbQ.count
            + (wordEmbScale.count
                + posEmbScale.count
                + typeEmbScale.count
                + embLnWeight.count
                + embLnBias.count) * 4
        for l in layers {
            for lin in [l.qkv, l.attnOut, l.ffnInter, l.ffnOut] {
                total += lin.wQ.count + (lin.wScale.count + lin.bias.count + lin.invSmooth.count) * 4
            }
            if let q = l.qOnly, let kv = l.kvOnly {
                for lin in [q, kv] {
                    total += lin.wQ.count + (lin.wScale.count + lin.bias.count + lin.invSmooth.count) * 4
                }
            }
            total += (l.ln1Weight.count + l.ln1Bias.count + l.ln2Weight.count + l.ln2Bias.count) * 4
        }
        return total
    }
}

// MARK: - Binary Deserialization Helpers

@inline(__always)
private func readU64LE(_ base: UnsafePointer<UInt8>, cursor: inout Int, total: Int) throws -> Int {
    guard cursor + 8 <= total else { throw ArcticEmbedError.unexpectedEOF }
    let v = UnsafeRawPointer(base.advanced(by: cursor)).loadUnaligned(as: UInt64.self)
    cursor += 8
    return Int(UInt64(littleEndian: v))
}

private func readI8Vec(_ base: UnsafePointer<UInt8>, cursor: inout Int, total: Int) throws -> [Int8] {
    let len = try readU64LE(base, cursor: &cursor, total: total)
    guard cursor + len <= total else { throw ArcticEmbedError.unexpectedEOF }
    var out = [Int8](repeating: 0, count: len)
    if len > 0 {
        out.withUnsafeMutableBytes { dst in
            dst.baseAddress!.copyMemory(from: UnsafeRawPointer(base.advanced(by: cursor)), byteCount: len)
        }
    }
    cursor += len
    return out
}

private func readF32Vec(_ base: UnsafePointer<UInt8>, cursor: inout Int, total: Int) throws -> [Float] {
    let len = try readU64LE(base, cursor: &cursor, total: total)
    let byteCount = len * 4
    guard cursor + byteCount <= total else { throw ArcticEmbedError.unexpectedEOF }
    var out = [Float](repeating: 0.0, count: len)
    if byteCount > 0 {
        out.withUnsafeMutableBytes { dst in
            dst.baseAddress!.copyMemory(from: UnsafeRawPointer(base.advanced(by: cursor)), byteCount: byteCount)
        }
    }
    cursor += byteCount
    return out
}

private func readLinear(_ base: UnsafePointer<UInt8>, cursor: inout Int, total: Int) throws -> QuantizedLinearRaw {
    let nOut = try readU64LE(base, cursor: &cursor, total: total)
    let kIn = try readU64LE(base, cursor: &cursor, total: total)
    let wQ = try readI8Vec(base, cursor: &cursor, total: total)
    let wScale = try readF32Vec(base, cursor: &cursor, total: total)
    let bias = try readF32Vec(base, cursor: &cursor, total: total)
    let invSmooth = try readF32Vec(base, cursor: &cursor, total: total)
    return QuantizedLinearRaw(
        nOut: nOut,
        kIn: kIn,
        wQ: wQ,
        wScale: wScale,
        bias: bias,
        invSmooth: invSmooth
    )
}
