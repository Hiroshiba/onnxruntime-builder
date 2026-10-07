# Unsigned generic ONNX Runtime WASM pthread build

This experimental branch builds **ONNX Runtime 1.23.2**, with Emscripten **4.0.8**,
SIMD and pthreads. It extends the static-library approach from
[VOICEVOX/onnxruntime-builder#131](https://github.com/VOICEVOX/onnxruntime-builder/pull/131)
(head `117593885cd2a66e9cf17b4059e6424d7ea528c9`). It uses only the public Microsoft
sources. There is no code signing, VOICEVOX production source access or GPU backend.

## Validate an existing build

This branch reuses a digest-pinned diagnostic archive from the original build.
It does not rebuild ONNX Runtime. The original compile/link succeeded, but its
smoke used an ordinary environment. Threaded WASM's default session options
require an environment configured with global thread pools.
This workflow relinks the smoke with that corrected initialization, runs it,
and packages the original static archive only after successful validation.

The exact original build and successful validation commits/run IDs are recorded
separately in `BUILD_INFO.json`. A failed original build is never described as
successful. A separate, reviewed publication workflow can publish the resulting
verified archive without rebuilding it.

## Artifact

`onnxruntime-wasm-static-simd-threaded-1.23.2.tgz` contains:

- `lib/libonnxruntime_webassembly.a` and `include/onnxruntime/`
- `BUILD_INFO.json`: exact source/builder commits, compiler version and flags
- `SMOKE_TEST.txt`: successful Node.js pthread and ONNX inference check
- `SHA256SUMS`: checksum of the static archive
- Upstream license, third-party notices and version information

A companion `.tgz.sha256` verifies the compressed archive. Its distinct name avoids
confusing it with PR #131's SIMD **single-threaded** `onnxruntime-wasm-static` package.
This is a static link input, not an `onnxruntime-web` JavaScript/WASM distribution.

## Application link contract

Use Emscripten **4.0.8** throughout the native/WASM dependency graph and final link.
Compile native dependencies and link the browser glue with:

```text
-pthread -msimd128 -fwasm-exceptions
```

Rust code should enable the Emscripten atomics/bulk-memory target features and pass
`-pthread` to the final Emscripten linker. A library built without pthreads cannot be
made threaded merely by changing the number of threads in a session option.

- Preallocate enough workers with `-sPTHREAD_POOL_SIZE=...` before synchronously
  creating ORT environments; account for all live global pools and application workers
- **Create the environment with global thread pools**. In ORT 1.23.2, threaded
  WASM defaults to `use_per_session_threads=false`. Use
  `CreateEnvWithGlobalThreadPools` (or `Ort::Env` with `Ort::ThreadingOptions`),
  set global intra-op threads explicitly, and set global inter-op threads to one
- Call `DisablePerSessionThreads` on sessions and keep execution sequential.
  Setting only the session's intra-op thread count does not configure the global
  pool. An ordinary environment causes session construction to fail
- One shared global pool serves all sessions in that environment; size the worker
  pool for the actual live global pools and other application workers
- Run synchronous CORE operations in a dedicated browser Worker and keep its event
  loop available for worker initialization; load every generated glue/worker file
- Serve over localhost or HTTPS with `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`; require `crossOriginIsolated` and
  `SharedArrayBuffer` before starting the threaded mode
- Choose adequate main/pthread stacks, initial memory and maximum memory at the
  application link step. The static archive cannot encode those runtime choices
- Keep a separate non-pthread artifact for a true single-threaded browser baseline;
  there is no single-binary fallback between pthread and non-pthread builds

The CI smoke test checks final linking, shared-memory pthread create/join, ORT
session construction with two intra-op threads, and inference correctness. It does
not measure CPU utilization or prove a speedup, and it does not replace the browser
VOICEVOX CORE benchmark. No performance numbers are inferred from build success.

Sources: [ONNX Runtime build options](https://onnxruntime.ai/docs/build/web.html),
[Emscripten pthread requirements](https://emscripten.org/docs/porting/pthreads.html).
