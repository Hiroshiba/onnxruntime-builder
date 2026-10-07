#!/usr/bin/env bash
# Run from the GitHub workspace containing builder/ and onnxruntime/ checkouts.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_SHA:?}"
: "${ONNXRUNTIME_VERSION:?}"
: "${ONNXRUNTIME_COMMIT:?}"
: "${EMSCRIPTEN_VERSION:?}"

builder="$GITHUB_WORKSPACE/builder"
ort="$GITHUB_WORKSPACE/onnxruntime"
artifact="$GITHUB_WORKSPACE/artifact"
cd "$ort"
test "$(git rev-parse HEAD)" = "$ONNXRUNTIME_COMMIT"
test "$(tr -d '\r\n' < VERSION_NUMBER)" = "$ONNXRUNTIME_VERSION"

cd cmake/external/emsdk
./emsdk install "$EMSCRIPTEN_VERSION"
./emsdk activate "$EMSCRIPTEN_VERSION"
# shellcheck disable=SC1091
source ./emsdk_env.sh
cd "$ort"

# Match the native WebAssembly exception ABI used by Rust's Emscripten target.
# ONNX Runtime's enable_wasm_threads adds -pthread to C and C++ dependencies too.
export EMCC_CFLAGS='-fwasm-exceptions'
python ./tools/ci_build/build.py \
  --build_dir ./build \
  --config Release \
  --update --build --parallel 2 \
  --build_wasm_static_lib \
  --enable_wasm_simd \
  --enable_wasm_threads \
  --disable_wasm_exception_catching \
  --disable_rtti \
  --skip_tests

# Fail instead of silently publishing the single-threaded/default build.
for option in BUILD_WEBASSEMBLY_STATIC_LIB ENABLE_WEBASSEMBLY_SIMD ENABLE_WEBASSEMBLY_THREADS; do
  # ORT declares some flags as options, but leaves SIMD as an untyped -D entry.
  grep -Ex "onnxruntime_$option:(BOOL|UNINITIALIZED)=ON" build/Release/CMakeCache.txt
done
test -s build/Release/libonnxruntime_webassembly.a

# The static library has no JS glue of its own. Exercise a real final link,
# pthread create/join, ORT session construction (2 intra-op threads) and inference.
mkdir -p build/smoke
em++ "$builder/tests/wasm-threaded-smoke.cc" \
  -I include/onnxruntime/core/session \
  build/Release/libonnxruntime_webassembly.a \
  -O2 -pthread -msimd128 -fwasm-exceptions \
  -sPTHREAD_POOL_SIZE=4 \
  -sDEFAULT_PTHREAD_STACK_SIZE=2097152 \
  -sSTACK_SIZE=5242880 \
  -sINITIAL_MEMORY=134217728 \
  -sALLOW_MEMORY_GROWTH=1 \
  -sEXIT_RUNTIME=1 \
  -sENVIRONMENT=node \
  --embed-file onnxruntime/test/testdata/mul_1.onnx@/mul_1.onnx \
  -o build/smoke/wasm-threaded-smoke.js
timeout 120 node build/smoke/wasm-threaded-smoke.js | tee build/smoke/result.txt

mkdir -p "$artifact/lib" "$artifact/include"
cp build/Release/libonnxruntime_webassembly.a "$artifact/lib/"
cp -R include/onnxruntime "$artifact/include/"
cp LICENSE README.md ThirdPartyNotices.txt VERSION_NUMBER "$artifact/"
cp "$builder/WASM_THREADED.md" "$artifact/"
git rev-parse HEAD > "$artifact/GIT_COMMIT_ID"
cp build/smoke/result.txt "$artifact/SMOKE_TEST.txt"
jq -n \
  --arg version "$ONNXRUNTIME_VERSION" \
  --arg source_commit "$ONNXRUNTIME_COMMIT" \
  --arg builder_commit "$GITHUB_SHA" \
  --arg emscripten_version "$EMSCRIPTEN_VERSION" \
  --arg node_version "$(node --version)" \
  '{
    schema_version: 1,
    library: "onnxruntime",
    version: $version,
    source_commit: $source_commit,
    builder_commit: $builder_commit,
    emscripten_version: $emscripten_version,
    target: "wasm32-unknown-emscripten",
    simd: true,
    pthreads: true,
    signed: false,
    exception_abi: "wasm",
    build_flags: ["--build_wasm_static_lib", "--enable_wasm_simd", "--enable_wasm_threads", "--disable_wasm_exception_catching", "--disable_rtti"],
    required_link_flags: ["-pthread", "-msimd128", "-fwasm-exceptions"],
    smoke_test: {passed: true, runtime: "node", runtime_version: $node_version, intra_op_threads: 2}
  }' > "$artifact/BUILD_INFO.json"
(
  cd "$artifact"
  sha256sum lib/libonnxruntime_webassembly.a > SHA256SUMS
)
