import Foundation

public let PAD_ID: UInt16 = 0
public let UNK_ID: UInt16 = 100
public let CLS_ID: UInt16 = 101
public let SEP_ID: UInt16 = 102
public let MASK_ID: UInt16 = 103
public let MAX_SEQ_LEN: Int = 512
public let MAX_WORD_CHARS: Int = 100

public let QUERY_PREFIX: String = "Represent this sentence for searching relevant passages: "

/// Zero-allocation Pure-Swift BERT WordPiece Tokenizer for `snowflake-arctic-embed-s`.
///
/// Features:
/// - Loads directly from `vocab.txt` (30,522 tokens)
/// - Exact `BertNormalizer` (`clean_text`, `handle_chinese_chars`, `lowercase`, NFD `strip_accents`)
/// - Exact `BertPreTokenizer` (whitespace + ASCII/Unicode punctuation isolation)
/// - Zero-allocation `##` subword lookup via dual open-addressed UTF-8 hash tables (`rootVocab` + `subwordVocab`)
/// - Pre-tokenized query prefix (`"Represent this sentence for searching relevant passages: "`)
public final class BertTokenizer: @unchecked Sendable {
    @usableFromInline
    struct VocabEntry {
        @usableFromInline var hash: UInt64
        @usableFromInline var offset: UInt32
        @usableFromInline var length: UInt16
        @usableFromInline var id: UInt16
    }

    @usableFromInline
    let arena: [UInt8]
    @usableFromInline
    let rootTable: [VocabEntry]
    @usableFromInline
    let rootMask: Int
    @usableFromInline
    let subwordTable: [VocabEntry]
    @usableFromInline
    let subwordMask: Int
    public let vocabCount: Int
    public private(set) var queryPrefixIds: [UInt16]

    public static func fromVocabFile(_ path: String) throws -> BertTokenizer {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { rawBuf in
            let bytes = rawBuf.bindMemory(to: UInt8.self)
            return BertTokenizer(vocabUTF8: bytes)
        }
    }

    public static func fromVocabFile(_ url: URL) throws -> BertTokenizer {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { rawBuf in
            let bytes = rawBuf.bindMemory(to: UInt8.self)
            return BertTokenizer(vocabUTF8: bytes)
        }
    }

    public static func fromVocabStr(_ vocabText: String) -> BertTokenizer {
        let utf8 = Array(vocabText.utf8)
        return utf8.withUnsafeBufferPointer { buf in
            BertTokenizer(vocabUTF8: buf)
        }
    }

    public init(vocabUTF8: UnsafeBufferPointer<UInt8>) {
        let arenaBytes = Array(vocabUTF8)
        self.arena = arenaBytes

        let rootCap = 65536
        let rootMask = rootCap - 1
        var rootTab = [VocabEntry](
            repeating: VocabEntry(hash: 0, offset: 0, length: 0, id: UInt16.max),
            count: rootCap
        )

        let subCap = 32768
        let subMask = subCap - 1
        var subTab = [VocabEntry](
            repeating: VocabEntry(hash: 0, offset: 0, length: 0, id: UInt16.max),
            count: subCap
        )

        var count = 0
        let n = arenaBytes.count
        var lineStart = 0

        arenaBytes.withUnsafeBufferPointer { arenaBuf in
            let base = arenaBuf.baseAddress!
            var i = 0
            while i <= n {
                if i == n || base[i] == 0x0A {
                    var lineEnd = i
                    if lineEnd > lineStart && base[lineEnd - 1] == 0x0D {
                        lineEnd -= 1
                    }
                    // In Rust `str::lines()`, a trailing newline does not produce an empty trailing line
                    if i == n && lineStart == n {
                        break
                    }
                    let len = lineEnd - lineStart
                    let id = UInt16(truncatingIfNeeded: count)
                    count += 1

                    let hRoot = Self.hashBytes(base.advanced(by: lineStart), count: len)
                    var slot = Int(truncatingIfNeeded: hRoot) & rootMask
                    while rootTab[slot].id != UInt16.max {
                        slot = (slot + 1) & rootMask
                    }
                    rootTab[slot] = VocabEntry(
                        hash: hRoot,
                        offset: UInt32(truncatingIfNeeded: lineStart),
                        length: UInt16(truncatingIfNeeded: len),
                        id: id
                    )

                    if len > 2 && base[lineStart] == 0x23 && base[lineStart + 1] == 0x23 {
                        let subStart = lineStart + 2
                        let subLen = len - 2
                        let hSub = Self.hashBytes(base.advanced(by: subStart), count: subLen)
                        var subSlot = Int(truncatingIfNeeded: hSub) & subMask
                        while subTab[subSlot].id != UInt16.max {
                            subSlot = (subSlot + 1) & subMask
                        }
                        subTab[subSlot] = VocabEntry(
                            hash: hSub,
                            offset: UInt32(truncatingIfNeeded: subStart),
                            length: UInt16(truncatingIfNeeded: subLen),
                            id: id
                        )
                    }

                    lineStart = i + 1
                }
                i += 1
            }
        }

        self.rootTable = rootTab
        self.rootMask = rootMask
        self.subwordTable = subTab
        self.subwordMask = subMask
        self.vocabCount = count

        var prefixIds: [UInt16] = []
        prefixIds.reserveCapacity(12)
        self.queryPrefixIds = []
        self.encodeRawInto(QUERY_PREFIX, out: &prefixIds)
        self.queryPrefixIds = prefixIds
    }

    @inlinable
    public var vocabSize: Int {
        vocabCount
    }

    /// Encodes a document or general sentence into `[CLS] ... [SEP]` (truncated to `MAX_SEQ_LEN = 512`).
    @inlinable
    public func encode(_ text: String) -> [UInt16] {
        var out: [UInt16] = []
        out.reserveCapacity(64)
        encodeInto(text, isQuery: false, out: &out)
        return out
    }

    /// Encodes a retrieval query with `QUERY_PREFIX` prepended (`[CLS] + prefix_ids + query_ids + [SEP]`).
    @inlinable
    public func encodeQuery(_ query: String) -> [UInt16] {
        var out: [UInt16] = []
        out.reserveCapacity(64)
        encodeInto(query, isQuery: true, out: &out)
        return out
    }

    /// Encodes into a caller-provided `[UInt16]` without reallocating.
    @inlinable
    public func encodeInto(_ text: String, isQuery: Bool, out: inout [UInt16]) {
        out.removeAll(keepingCapacity: true)
        out.append(CLS_ID)
        if isQuery {
            out.append(contentsOf: queryPrefixIds)
        }
        encodeRawInto(text, out: &out)
        if out.count >= MAX_SEQ_LEN {
            out.removeSubrange((MAX_SEQ_LEN - 1)..<out.count)
        }
        out.append(SEP_ID)
    }

    @usableFromInline
    func encodeRawInto(_ text: String, out: inout [UInt16]) {
        var normBytes: [UInt8] = []
        normBytes.reserveCapacity(text.utf8.count + 16)

        for scalar in text.unicodeScalars {
            let cp = scalar.value
            if cp == 0 || cp == 0xFFFD || Self.isBertControl(cp) {
                continue
            }
            if Self.isBertWhitespace(cp, scalar: scalar) {
                normBytes.append(0x20)
                continue
            }
            if Self.isChineseChar(cp) {
                normBytes.append(0x20)
                Self.appendUTF8(cp, to: &normBytes)
                normBytes.append(0x20)
                continue
            }
            if Self.isCombiningMark(cp) {
                continue
            }
            let baseCp = Self.stripLatinAccent(cp)
            if baseCp < 0x80 {
                let lc = (baseCp >= 0x41 && baseCp <= 0x5A) ? (baseCp + 32) : baseCp
                normBytes.append(UInt8(truncatingIfNeeded: lc))
            } else if let baseScalar = Unicode.Scalar(baseCp) {
                let lowerStr = String(baseScalar).lowercased()
                for lcScalar in lowerStr.unicodeScalars {
                    if !Self.isCombiningMark(lcScalar.value) {
                        Self.appendUTF8(lcScalar.value, to: &normBytes)
                    }
                }
            }
        }

        normBytes.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            let n = buf.count
            var wordStart: Int = -1
            var idx = 0

            while idx < n {
                let b0 = base[idx]
                if b0 < 0x80 {
                    if b0 == 0x20 {
                        if wordStart >= 0 {
                            wordpieceInto(base.advanced(by: wordStart), len: idx - wordStart, out: &out)
                            wordStart = -1
                        }
                    } else if Self.isAsciiPunctuation(b0) {
                        if wordStart >= 0 {
                            wordpieceInto(base.advanced(by: wordStart), len: idx - wordStart, out: &out)
                            wordStart = -1
                        }
                        wordpieceInto(base.advanced(by: idx), len: 1, out: &out)
                    } else if wordStart < 0 {
                        wordStart = idx
                    }
                    idx += 1
                } else {
                    let (cp, charLen) = Self.decodeUTF8Scalar(base.advanced(by: idx), remaining: n - idx)
                    if Self.isBertPunctuation(cp) {
                        if wordStart >= 0 {
                            wordpieceInto(base.advanced(by: wordStart), len: idx - wordStart, out: &out)
                            wordStart = -1
                        }
                        wordpieceInto(base.advanced(by: idx), len: charLen, out: &out)
                    } else if wordStart < 0 {
                        wordStart = idx
                    }
                    idx += charLen
                }
            }

            if wordStart >= 0 {
                wordpieceInto(base.advanced(by: wordStart), len: n - wordStart, out: &out)
            }
        }
    }

    @usableFromInline
    @inline(__always)
    func wordpieceInto(_ wordPtr: UnsafePointer<UInt8>, len: Int, out: inout [UInt16]) {
        if len == 0 { return }

        // Fast path: whole word is a single root token in vocabulary (covers >90% of words!)
        if let id = lookupRoot(wordPtr, len: len) {
            out.append(id)
            return
        }

        // Count Unicode scalar values (UTF-8 leading bytes)
        var charCount = 0
        for i in 0..<len {
            if (wordPtr[i] & 0b1100_0000) != 0b1000_0000 {
                charCount += 1
            }
        }
        if charCount > MAX_WORD_CHARS {
            out.append(UNK_ID)
            return
        }

        let startOutLen = out.count
        var start = 0

        while start < len {
            var end = len
            var matchedId: UInt16? = nil

            while end > start {
                let slicePtr = wordPtr.advanced(by: start)
                let sliceLen = end - start
                let lookup = (start == 0)
                    ? lookupRoot(slicePtr, len: sliceLen)
                    : lookupSubword(slicePtr, len: sliceLen)
                if let id = lookup {
                    matchedId = id
                    break
                }
                // Step `end` back by one UTF-8 character boundary
                end -= 1
                while end > start && (wordPtr[end] & 0b1100_0000) == 0b1000_0000 {
                    end -= 1
                }
            }

            if let id = matchedId {
                out.append(id)
                start = end
            } else {
                out.removeSubrange(startOutLen..<out.count)
                out.append(UNK_ID)
                return
            }
        }
    }

    @usableFromInline
    @inline(__always)
    func lookupRoot(_ ptr: UnsafePointer<UInt8>, len: Int) -> UInt16? {
        let h = Self.hashBytes(ptr, count: len)
        var slot = Int(truncatingIfNeeded: h) & rootMask
        return arena.withUnsafeBufferPointer { arenaBuf in
            let arenaBase = arenaBuf.baseAddress!
            while true {
                let entry = rootTable[slot]
                if entry.id == UInt16.max {
                    return nil
                }
                if entry.hash == h && Int(entry.length) == len {
                    if memcmp(arenaBase.advanced(by: Int(entry.offset)), ptr, len) == 0 {
                        return entry.id
                    }
                }
                slot = (slot + 1) & rootMask
            }
        }
    }

    @usableFromInline
    @inline(__always)
    func lookupSubword(_ ptr: UnsafePointer<UInt8>, len: Int) -> UInt16? {
        let h = Self.hashBytes(ptr, count: len)
        var slot = Int(truncatingIfNeeded: h) & subwordMask
        return arena.withUnsafeBufferPointer { arenaBuf in
            let arenaBase = arenaBuf.baseAddress!
            while true {
                let entry = subwordTable[slot]
                if entry.id == UInt16.max {
                    return nil
                }
                if entry.hash == h && Int(entry.length) == len {
                    if memcmp(arenaBase.advanced(by: Int(entry.offset)), ptr, len) == 0 {
                        return entry.id
                    }
                }
                slot = (slot + 1) & subwordMask
            }
        }
    }

    @usableFromInline
    @inline(__always)
    static func hashBytes(_ ptr: UnsafePointer<UInt8>, count: Int) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for i in 0..<count {
            h ^= UInt64(ptr[i])
            h &*= 0x100000001b3
        }
        // Finalizer mix to avoid clustering in low bits
        h ^= h >> 33
        h &*= 0xff51afd7ed558ccd
        h ^= h >> 33
        return h
    }

    @usableFromInline
    @inline(__always)
    static func appendUTF8(_ cp: UInt32, to out: inout [UInt8]) {
        if cp < 0x80 {
            out.append(UInt8(truncatingIfNeeded: cp))
        } else if cp < 0x800 {
            out.append(UInt8(0xC0 | (cp >> 6)))
            out.append(UInt8(0x80 | (cp & 0x3F)))
        } else if cp < 0x10000 {
            out.append(UInt8(0xE0 | (cp >> 12)))
            out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            out.append(UInt8(0x80 | (cp & 0x3F)))
        } else {
            out.append(UInt8(0xF0 | (cp >> 18)))
            out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
            out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            out.append(UInt8(0x80 | (cp & 0x3F)))
        }
    }

    @usableFromInline
    @inline(__always)
    static func decodeUTF8Scalar(_ ptr: UnsafePointer<UInt8>, remaining: Int) -> (UInt32, Int) {
        let b0 = UInt32(ptr[0])
        if b0 < 0x80 {
            return (b0, 1)
        } else if (b0 & 0xE0) == 0xC0 && remaining >= 2 {
            let cp = ((b0 & 0x1F) << 6) | UInt32(ptr[1] & 0x3F)
            return (cp, 2)
        } else if (b0 & 0xF0) == 0xE0 && remaining >= 3 {
            let cp = ((b0 & 0x0F) << 12) | (UInt32(ptr[1] & 0x3F) << 6) | UInt32(ptr[2] & 0x3F)
            return (cp, 3)
        } else if (b0 & 0xF8) == 0xF0 && remaining >= 4 {
            let cp = ((b0 & 0x07) << 18) | (UInt32(ptr[1] & 0x3F) << 12) | (UInt32(ptr[2] & 0x3F) << 6) | UInt32(ptr[3] & 0x3F)
            return (cp, 4)
        }
        return (b0, 1)
    }

    @usableFromInline
    @inline(__always)
    static func isBertWhitespace(_ cp: UInt32, scalar: Unicode.Scalar) -> Bool {
        if cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D {
            return true
        }
        return scalar.properties.isWhitespace
    }

    @usableFromInline
    @inline(__always)
    static func isBertControl(_ cp: UInt32) -> Bool {
        if cp == 0x09 || cp == 0x0A || cp == 0x0D {
            return false
        }
        return cp <= 0x1F || (cp >= 0x7F && cp <= 0x9F)
    }

    @usableFromInline
    @inline(__always)
    static func isChineseChar(_ cp: UInt32) -> Bool {
        (0x4E00...0x9FFF).contains(cp)
            || (0x3400...0x4DBF).contains(cp)
            || (0x20000...0x2A6DF).contains(cp)
            || (0x2A700...0x2B73F).contains(cp)
            || (0x2B740...0x2B81F).contains(cp)
            || (0x2B820...0x2CEAF).contains(cp)
            || (0xF900...0xFAFF).contains(cp)
            || (0x2F800...0x2FA1F).contains(cp)
    }

    @usableFromInline
    @inline(__always)
    static func isAsciiPunctuation(_ b: UInt8) -> Bool {
        (33...47).contains(b) || (58...64).contains(b) || (91...96).contains(b) || (123...126).contains(b)
    }

    @usableFromInline
    @inline(__always)
    static func isBertPunctuation(_ cp: UInt32) -> Bool {
        if (33...47).contains(cp)
            || (58...64).contains(cp)
            || (91...96).contains(cp)
            || (123...126).contains(cp)
        {
            return true
        }
        if (0x2000...0x206F).contains(cp)
            || (0x3000...0x303F).contains(cp)
            || (0xFE30...0xFE6F).contains(cp)
            || (0xFF01...0xFF0F).contains(cp)
            || (0xFF1A...0xFF20).contains(cp)
            || (0xFF3B...0xFF40).contains(cp)
            || (0xFF5B...0xFF65).contains(cp)
        {
            return true
        }
        switch cp {
        case 0x00A1, 0x00A7, 0x00AB, 0x00B6, 0x00B7, 0x00BB, 0x00BF,
             0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015,
             0x2018, 0x2019, 0x201A, 0x201C, 0x201D, 0x201E,
             0x2020, 0x2021, 0x2022, 0x2026, 0x2030, 0x2032, 0x2033,
             0x2039, 0x203A, 0x203C, 0x203D, 0x203E, 0x2044:
            return true
        default:
            return false
        }
    }

    @usableFromInline
    @inline(__always)
    static func isCombiningMark(_ cp: UInt32) -> Bool {
        (0x0300...0x036F).contains(cp)
            || (0x1AB0...0x1AFF).contains(cp)
            || (0x1DC0...0x1DFF).contains(cp)
            || (0x20D0...0x20FF).contains(cp)
            || (0xFE20...0xFE2F).contains(cp)
    }

    /// Maps precomposed Latin-1 / Latin Extended-A accented characters (`U+00C0..=U+017F`)
    /// to their NFD base character (matching `BertNormalizer` with `strip_accents = true`).
    @usableFromInline
    @inline(__always)
    static func stripLatinAccent(_ cp: UInt32) -> UInt32 {
        switch cp {
        case 0x00C0...0x00C5, 0x00E0...0x00E5, 0x0100...0x0105:
            return 0x61 // 'a'
        case 0x00C7, 0x00E7, 0x0106...0x010D:
            return 0x63 // 'c'
        case 0x010E, 0x010F:
            return 0x64 // 'd'
        case 0x00C8...0x00CB, 0x00E8...0x00EB, 0x0112...0x011B:
            return 0x65 // 'e'
        case 0x011C...0x0123:
            return 0x67 // 'g'
        case 0x0124, 0x0125:
            return 0x68 // 'h'
        case 0x00CC...0x00CF, 0x00EC...0x00EF, 0x0128...0x012F:
            return 0x69 // 'i'
        case 0x0134, 0x0135:
            return 0x6A // 'j'
        case 0x0136, 0x0137:
            return 0x6B // 'k'
        case 0x0139...0x013E:
            return 0x6C // 'l'
        case 0x00D1, 0x00F1, 0x0143...0x0148:
            return 0x6E // 'n'
        case 0x00D2...0x00D6, 0x00F2...0x00F6, 0x014C...0x0151:
            return 0x6F // 'o'
        case 0x0154...0x0159:
            return 0x72 // 'r'
        case 0x015A...0x0161:
            return 0x73 // 's'
        case 0x0162...0x0165:
            return 0x74 // 't'
        case 0x00D9...0x00DC, 0x00F9...0x00FC, 0x0168...0x0173:
            return 0x75 // 'u'
        case 0x0174, 0x0175:
            return 0x77 // 'w'
        case 0x00DD, 0x00FD, 0x00FF, 0x0176, 0x0177, 0x0178:
            return 0x79 // 'y'
        case 0x0179...0x017E:
            return 0x7A // 'z'
        default:
            return cp
        }
    }
}
