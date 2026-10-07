#!/usr/bin/env bash
# Reuse the immutable compiled archive; only the small validation application links.
set -euo pipefail
: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_SHA:?}"
: "${GITHUB_RUN_ID:?}"
: "${SOURCE_BUILD_RUN_ID:?}"
: "${SOURCE_BUILD_COMMIT:?}"
: "${ONNXRUNTIME_COMMIT:?}"
: "${ONNXRUNTIME_VERSION:?}"
: "${EMSCRIPTEN_VERSION:?}"
: "${EXPECTED_LIBRARY_SHA256:?}"
: "${RELEASE_NAME:?}"
cd "$GITHUB_WORKSPACE"
builder="$GITHUB_WORKSPACE/builder"
raw="$GITHUB_WORKSPACE/unvalidated"
lib="$raw/build/Release/libonnxruntime_webassembly.a"
artifact="$GITHUB_WORKSPACE/$RELEASE_NAME"
printf '%s  %s\n' "$EXPECTED_LIBRARY_SHA256" "$lib" | sha256sum --check
for option in BUILD_WEBASSEMBLY_STATIC_LIB ENABLE_WEBASSEMBLY_SIMD ENABLE_WEBASSEMBLY_THREADS; do
  grep -Ex "onnxruntime_$option:(BOOL|UNINITIALIZED)=ON" "$raw/build/Release/CMakeCache.txt"
done
test "$(git -C onnxruntime rev-parse HEAD)" = "$ONNXRUNTIME_COMMIT"
test "$(tr -d '\r\n' < onnxruntime/VERSION_NUMBER)" = "$ONNXRUNTIME_VERSION"

./emsdk/emsdk install "$EMSCRIPTEN_VERSION"
./emsdk/emsdk activate "$EMSCRIPTEN_VERSION"
# shellcheck disable=SC1091
source ./emsdk/emsdk_env.sh
mkdir -p validation
# Match the original smoke link exactly; only the global-pool initialization changed.
em++ "$builder/tests/wasm-threaded-smoke.cc" \
  -I "$raw/include/onnxruntime/core/session" \
  "$lib" \
  -O2 -pthread -msimd128 -fwasm-exceptions \
  -sPTHREAD_POOL_SIZE=4 \
  -sDEFAULT_PTHREAD_STACK_SIZE=2097152 \
  -sSTACK_SIZE=5242880 \
  -sINITIAL_MEMORY=134217728 \
  -sALLOW_MEMORY_GROWTH=1 \
  -sEXIT_RUNTIME=1 \
  -sENVIRONMENT=node \
  --embed-file onnxruntime/onnxruntime/test/testdata/mul_1.onnx@/mul_1.onnx \
  -o validation/wasm-threaded-smoke.js
timeout 120 node validation/wasm-threaded-smoke.js | tee validation/result.txt

mkdir -p "$artifact/lib" "$artifact/include"
cp "$lib" "$artifact/lib/"
cp -R "$raw/include/onnxruntime" "$artifact/include/"
cp onnxruntime/{LICENSE,README.md,ThirdPartyNotices.txt,VERSION_NUMBER} "$artifact/"
cp "$builder/WASM_THREADED.md" "$artifact/"
printf '%s\n' "$ONNXRUNTIME_COMMIT" > "$artifact/GIT_COMMIT_ID"
cp validation/result.txt "$artifact/SMOKE_TEST.txt"
jq -n \
  --arg version "$ONNXRUNTIME_VERSION" --arg source_commit "$ONNXRUNTIME_COMMIT" \
  --arg builder_commit "$SOURCE_BUILD_COMMIT" --arg validation_commit "$GITHUB_SHA" \
  --argjson build_run_id "$SOURCE_BUILD_RUN_ID" --argjson validation_run_id "$GITHUB_RUN_ID" \
  --arg emscripten_version "$EMSCRIPTEN_VERSION" --arg node_version "$(node --version)" \
  '{
    schema_version: 1, library: "onnxruntime", version: $version,
    source_commit: $source_commit, builder_commit: $builder_commit,
    validation_commit: $validation_commit,
    build_run_id: $build_run_id, validation_run_id: $validation_run_id,
    emscripten_version: $emscripten_version, target: "wasm32-unknown-emscripten",
    simd: true, pthreads: true, signed: false, exception_abi: "wasm",
    build_flags: ["--build_wasm_static_lib", "--enable_wasm_simd", "--enable_wasm_threads", "--disable_wasm_exception_catching", "--disable_rtti"],
    required_link_flags: ["-pthread", "-msimd128", "-fwasm-exceptions"],
    thread_pool_scope: "global",
    smoke_test: {passed: true, runtime: "node", runtime_version: $node_version, intra_op_threads: 2, inter_op_threads: 1}
  }' > "$artifact/BUILD_INFO.json"
(
  cd "$artifact"
  sha256sum lib/libonnxruntime_webassembly.a > SHA256SUMS
)
tar -czf "$RELEASE_NAME.tgz" "$RELEASE_NAME"
sha256sum "$RELEASE_NAME.tgz" > "$RELEASE_NAME.tgz.sha256"
