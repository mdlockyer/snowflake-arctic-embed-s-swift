# snowflake-arctic-embed-s-swift

Pure-Swift Int8 CPU inference for [`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)
(33M-parameter BERT, 384-dim embeddings). No ONNX Runtime, no BLAS/Accelerate, no external dependencies: hand-tuned
SIMD kernels for macOS and Linux on ARM64 and x86_64.

- **W8A8 SmoothQuant** with per-row dynamic activation quantization; embeddings stay within
  cosine 0.9990 of the FP32 model (mean 0.9994 on short texts, 0.9993 on 124–512-token texts).
- **Runtime-selected kernels:** ARM64 SME2 (Apple M4 `smopa`), `FEAT_I8MM`, `DotProd`, NEON;
  x86_64 AVX-VNNI, AVX2; portable scalar fallback.
- **Latency mode:** one sequence split across both SME units of an M4 Pro/Max (`ParallelContext`).
- **Throughput mode:** multi-threaded ragged batching (`embedBatch`).
- **Fast startup:** the pre-quantized `.ae8` model pack (33.8 MB) loads and is packed for the host CPU in ~10 ms.

## Performance (Apple M4 Max, SME backend)

| Tokens | 1 thread | 2 threads (`ParallelContext`) |
|-------:|---------:|------------------------------:|
| 8      | 421 µs   | 332 µs |
| 32     | 1.23 ms  | 0.77 ms |
| 128    | 5.19 ms  | 3.14 ms |
| 512    | 32.6 ms  | 19.0 ms |

Reproduce with `swift run -c release arctic-embed` (add `--quick` for a fast smoke run with
~20x fewer iterations, as CI does).

## Quick start

```bash
scripts/fetch_model.sh              # fetches model/model.ae8 (v1.0.0, SHA-256 verified)
swift run -c release arctic-embed   # runs retrieval demo & benchmark suite
```

```swift
import ArcticEmbedS

let engine = try ArcticEmbedS.fromDir("model") // model.ae8 + vocab.txt
let ws = Workspace()
let q = engine.embedQuery("what is snowflake?", ws: ws) // adds the query prefix
let d = engine.embedDocument("Snowflake is a cloud data platform.", ws: ws)
print("similarity = \(cosineSimilarity(q, d))")

// Lowest single-query latency on SME machines (2 threads; single-threaded elsewhere):
let par = ParallelContext.recommended(engine)
let qPar = engine.embedQueryParallel("what is snowflake?", par: par)

// Throughput: many texts across all cores.
let docs = engine.embedBatch(["first doc", "second doc"], isQuery: false, numThreads: 8)
```

Embeddings are L2-normalized `[Float]` of length `384` taken from the `[CLS]` token, matching the reference
sentence-transformers pipeline. Inputs are truncated to 512 tokens.

## Model files

The library reads a model directory containing `model.ae8` (Int8 SmoothQuant weights) and `vocab.txt`.
`model.ae8` is backend-neutral and is built by [`snowflake-arctic-embed-s-weights`](https://github.com/mdlockyer/snowflake-arctic-embed-s-weights)
(also mirrored in [`snowflake-arctic-embed-s-rust`](https://github.com/mdlockyer/snowflake-arctic-embed-s-rust)).

Fetch the pinned release pack into `./model` with:

```bash
scripts/fetch_model.sh
```

## Testing

```bash
scripts/fetch_model.sh
swift test -c release
```

| Test | Checks |
|------|--------|
| `testTokenizerAndInt8AccuracyAgainstFp32Reference` | Tokenizer ids and embeddings vs. FP32 reference vectors; ragged batching vs. single-sequence |
| `testAllSimdBackendsMatchScalar` | Every SIMD backend the host supports vs. Scalar |
| `testLongSequenceAccuracyAgainstFp64Reference` | 124–512-token texts vs. a float64 reference |
| `testParallelForwardMatchesSingleThread` | `ParallelContext` output is bit-identical to the single-threaded forward pass (SME only) |

## Layout

| Path | Contents |
|------|----------|
| `Sources/ArcticEmbedS/Quantize.swift` | `.ae8` binary model loader (`QuantizedModelRaw`) |
| `Sources/ArcticEmbedS/Gemm.swift` | Backend detection and Int8 GEMM kernels (SME, I8MM, DotProd, NEON, AVX-VNNI, AVX2, Scalar) |
| `Sources/ArcticEmbedS/SimdMath.swift` | Attention, softmax, GELU, LayerNorm and quantization kernels |
| `Sources/ArcticEmbedS/AArch64AsmKernels.swift` | AArch64 SME2 / I8MM / FP16 / DotProd machine-code kernels |
| `Sources/ArcticEmbedS/X86AsmKernels.swift` | x86_64 CPUID + AVX2 / AVX-VNNI machine-code kernels |
| `Sources/ArcticEmbedS/Model.swift` | Per-backend packed model and single-sequence / ragged forward pass |
| `Sources/ArcticEmbedS/Parallel.swift` | Intra-op multi-core forward pass for one sequence |
| `Sources/ArcticEmbedS/Tokenizer.swift` | Zero-allocation BERT WordPiece tokenizer |
| `Sources/ArcticEmbedS/AlignedBuffer.swift` | 64-byte cache-line aligned buffer |
| `Sources/ArcticEmbedS/ArcticEmbedS.swift` | Top-level `ArcticEmbedS` engine and multi-threaded `embedBatch` |
| `Sources/arctic-embed/main.swift` | `arctic-embed`: CLI retrieval demo and benchmark suite |
| `scripts/fetch_model.sh` | Fetches `model.ae8` + `vocab.txt` from `snowflake-arctic-embed-s-weights` |

## Model

`snowflake-arctic-embed-s` is © Snowflake Inc., released under Apache-2.0. `model/model.ae8` is a
modified (Int8-quantized) derivative of its weights; see [`model/README.md`](model/README.md) and
[`NOTICE`](NOTICE).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project shall be licensed as above, without any additional terms or conditions.
