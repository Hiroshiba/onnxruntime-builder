# Unsigned FP32 XNNPACK WASM SIMD pthread experiment

This branch builds **ONNX Runtime 1.23.2** at
`a83fc4d58cb48eb68890dd689f94f28288cf2278` with **Emscripten 4.0.8**, ordinary
WASM SIMD, pthreads and the **XNNPACK execution provider**. It is a separate
experiment from the validated CPU-only SIMD pthread archive. No baseline release
or default branch is replaced, and no VOICEVOX production sources or signing
credentials are used.

## Build and distribution

The fork-only `build.yml` builds this one target on push. To also create a draft
prerelease, dispatch it on branch `feat/wasm-simd-pthreads-xnnpack` with
`version=1.23.2`, `target=onnxruntime`, `release=true`, `code_signing=false`.
An existing release is never overwritten. Cached Emscripten and native dependencies
are reused on equivalent builds. Unvalidated archive, compile commands, CMake cache
and smoke diagnostics are retained on failure.

The distinct archive is:

`onnxruntime-wasm-static-simd-threaded-xnnpack-1.23.2.tgz`

It contains the same application-facing `lib/libonnxruntime_webassembly.a` and
`include/onnxruntime/` layout as the CPU-only archive. XNNPACK and pthreadpool are
folded into the single archive. `xnnpack-static-dependencies.patch` makes the static
bundler traverse the provider's missing dependency edges. It also selects the
pinned pthreadpool's existing pthread/futex implementation when WASM threads are
enabled; upstream otherwise selects a serial Emscripten shim. A target-local
POSIX feature definition exposes posix_memalign. These are build integration
changes, not new numerical kernels or model execution code. ORT 1.23.2 expects amalgamation files that its pinned XNNPACK no longer ships.
The integration patch instead selects XNNPACK's generated production scalar,
WASM and ordinary-SIMD source lists, retains its lookup tables, and includes
`microkernels-prod` inside the final bundled archive. All kernel source is unchanged.
Only the archive bundling target is built. CMake install rules are disabled because
this distribution copies the archive/headers directly and does not export native
CMake packages.

Provenance and evidence included:

- `BUILD_INFO.json`: compiler/source/builder pins, flags, EP and thread contract
- `COMPILE_AUDIT.json`: configured C/C++ compile-command checks
- `PTHREADPOOL_SMOKE.txt`: two-thread work dispatch verified before the full build
- `XNNPACK_SMOKE.json`: actual provider assignments, profile and worker counts
- `SMOKE_TEST.txt`: complete final-link smoke output
- `SHA256SUMS`: archive/evidence/patch file checksums
- Upstream licenses, third-party notices and the exact source patch

XNNPACK is pinned by ORT to `fe98e0b93565382648129271381c14d6205255e3` and pthreadpool
to `4e80ca24521aa0fb3a746f9ea9c3eaa20e9afbb0`; their downloads retain upstream hash
verification. Tests and unused non-production XNNPACK microkernels are not built.

## Floating-point scope

The experiment runs unchanged FP32 model tensors. It does not quantize, convert to
FP16, enable relaxed SIMD, or enable fast math. Both compile and final-link flags
include `-fno-fast-math -ffp-contract=off`. CMake disables relaxed SIMD and ORT CPU
FP16 training ops. The generic runtime still contains support for other ONNX data
types; this is not an FP32-only operator-stripped runtime.

The audit rejects `-ffast-math`, `-Ofast`, `-mrelaxed-simd`, `-mfp16`,
`-ffinite-math-only`, unsafe math, associative math and any relaxed-SIMD source file.
It requires pthread, SIMD, exception and strict math flags on every C/C++ unit,
requires the real pthreadpool sources and rejects the serial shim.
Changing provider may change accumulation order. Bit-identical VOICEVOX PCM is not
promised; measure PCM error against the matching CPU control.

## Application link and session contract

Use Emscripten **4.0.8** throughout, and link with:

```text
-pthread -msimd128 -fwasm-exceptions -fno-fast-math -ffp-contract=off
```

Register `XNNPACK` explicitly through `SessionOptionsAppendExecutionProvider`, with
provider option `intra_op_num_threads` set to the intended budget. Registration
must fail loudly when unavailable; merely having the EP compiled does not select it.

For the initial two-thread experiment:

- ORT environment global intra-op = **1**, global inter-op = **1**, spinning off
- Sessions disable per-session ORT pools; execution mode remains sequential
- XNNPACK intra-op = **2**, giving one XNNPACK worker plus the calling thread
- Register XNNPACK only on the measured decode session; unrelated model sessions
  must not allocate additional idle XNNPACK pools
- CPU fallback nodes run on the calling thread, so their cost is included
- Compare with a CPU-only session on the same fixed-shape graph and same input
  using ORT global intra-op = **2**

XNNPACK and ORT own different pools. Changing the session intra-op setting alone
cannot configure threaded WASM's global ORT pool. Account for all live XNNPACK
sessions when preallocating `PTHREAD_POOL_SIZE`; each EP instance owns its pool.
The smoke application preallocates four reusable workers and proves that only one
compute pthread is allocated by the tested XNNPACK session, with no ORT workers.
Preallocated but unused Emscripten Workers are not additional compute pools.

Run synchronous CORE work in a dedicated browser Worker. Serve localhost/HTTPS
with COOP `same-origin` and COEP `require-corp`; require `crossOriginIsolated` and
`SharedArrayBuffer`. Choose adequate stacks/memory at final link. This static
archive is not an `onnxruntime-web` JS/WASM package and has no non-pthread fallback.

## What the validation proves

The smoke builds a tiny fixed-shape FP32 graph with constant weights and bias:
Conv1d followed by ConvTranspose1d. CPU and XNNPACK outputs are independently checked
against scalar reference math. A final link uses only the bundled archive and C++
public headers. Profiling must assign **both** Conv and ConvTranspose to
`XnnpackExecutionProvider`; silent CPU fallback fails validation. A linker wrapper
counts successful `pthread_create` calls, first validating itself with create/join,
then requiring zero ORT worker creations and exactly one XNNPACK worker creation.
A separate small pthreadpool preflight checks that both caller and worker actually
execute tasks before the costly ORT build starts. WASM profile JSON is captured
from stdout, which is where the pinned ORT profiler writes it.

It does not prove the actual VOICEVOX decode graph is covered, that workers are
busy, or that XNNPACK is faster. VOICEVOX needs separate browser provider profiling,
output comparison and timing. In ORT 1.23.2, Conv1d/ConvTranspose1d eligibility
requires known channel/spatial shapes and constant weights. The dynamic sample
model therefore requires a separately labeled query-derived fixed-shape experiment,
with an equivalent CPU-only fixed-shape control. Do not label a fallback-only run
as XNNPACK acceleration.

## Primary source references

- [ORT static bundler](https://github.com/microsoft/onnxruntime/blob/v1.23.2/cmake/onnxruntime_webassembly.cmake)
- [ORT XNNPACK WASM configuration](https://github.com/microsoft/onnxruntime/blob/v1.23.2/cmake/external/xnnpack.cmake)
- [Conv1d shape eligibility](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/providers/xnnpack/nn/conv_base.cc)
- [Separate XNNPACK threadpool](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/providers/xnnpack/xnnpack_execution_provider.cc)
