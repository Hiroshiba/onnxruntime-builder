# Unsigned generic ONNX Runtime WASM pthread build

This experimental branch builds **ONNX Runtime 1.23.2**, with Emscripten **4.0.8**,
SIMD and pthreads. It extends the static-library approach from
[VOICEVOX/onnxruntime-builder#131](https://github.com/VOICEVOX/onnxruntime-builder/pull/131)
(head `117593885cd2a66e9cf17b4059e6424d7ea528c9`). It uses only the public Microsoft
sources. There is no code signing, VOICEVOX production source access or GPU backend.

## Run

This branch specializes the existing `build.yml`; the default branch is unchanged.
The existing `push` and `workflow_dispatch` triggers are retained. A push builds and
validates one Linux-hosted WASM artifact. To additionally create a draft prerelease:

1. Open [the fork's build workflow](https://github.com/Hiroshiba/onnxruntime-builder/actions/workflows/build.yml)
2. Choose **Run workflow**, branch **feat/wasm-simd-pthreads-benchmark**
3. Set `version=1.23.2`, `target=onnxruntime`, `release=true`, `code_signing=false`
4. Run it; review the smoke-test result and draft prerelease before publishing

The workflow retains the default branch's input names so it can be dispatched
without installing a new workflow on `main`. It rejects other versions, private
runtime targets and signing. It only executes in `Hiroshiba/onnxruntime-builder`.
An existing release is never overwritten.

The equivalent command, from an authenticated GitHub CLI, is:

```sh
gh workflow run build.yml --repo Hiroshiba/onnxruntime-builder \
  --ref feat/wasm-simd-pthreads-benchmark \
  -f version=1.23.2 -f target=onnxruntime -F release=true -F code_signing=false
```

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
  creating ORT sessions; account for simultaneous pools from all live sessions
- Set `intra_op_num_threads` explicitly for the benchmark; keep inter-op execution
  sequential unless that is a separate experimental variable
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
