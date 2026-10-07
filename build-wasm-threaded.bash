#!/usr/bin/env bash
# Build the distinct XNNPACK experiment from pinned public sources.
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
python "$builder/tests/test-xnnpack-verifier.py"

git apply --check "$builder/xnnpack-static-dependencies.patch"
git apply "$builder/xnnpack-static-dependencies.patch"

cd cmake/external/emsdk
./emsdk install "$EMSCRIPTEN_VERSION"
./emsdk activate "$EMSCRIPTEN_VERSION"
# shellcheck disable=SC1091
source ./emsdk_env.sh
cd "$ort"

# Match the native WebAssembly exception ABI used by Rust's Emscripten target.
# ONNX Runtime's enable_wasm_threads adds -pthread to C and C++ dependencies too.
export EMCC_CFLAGS='-fwasm-exceptions -fno-fast-math -ffp-contract=off'
python ./tools/ci_build/build.py \
  --build_dir ./build \
  --config Release \
  --update --parallel 2 \
  --build_wasm_static_lib \
  --enable_wasm_simd \
  --enable_wasm_threads \
  --use_xnnpack \
  --disable_wasm_exception_catching \
  --disable_rtti \
  --skip_tests \
  --cmake_extra_defines \
    CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    CMAKE_SKIP_INSTALL_RULES=ON \
    'CMAKE_C_FLAGS=-fwasm-exceptions -fno-fast-math -ffp-contract=off' \
    'CMAKE_CXX_FLAGS=-fwasm-exceptions -fno-fast-math -ffp-contract=off' \
    "FETCHCONTENT_BASE_DIR=$GITHUB_WORKSPACE/dependency-cache" \
    onnxruntime_ENABLE_WEBASSEMBLY_RELAXED_SIMD=OFF \
    onnxruntime_ENABLE_CPU_FP16_OPS=OFF \
    XNNPACK_BUILD_ALL_MICROKERNELS=OFF

# Fail instead of silently publishing the single-threaded/default build.
for option in BUILD_WEBASSEMBLY_STATIC_LIB ENABLE_WEBASSEMBLY_SIMD ENABLE_WEBASSEMBLY_THREADS; do
  # ORT declares some flags as options, but leaves SIMD as an untyped -D entry.
  grep -Ex "onnxruntime_$option:(BOOL|UNINITIALIZED)=ON" build/Release/CMakeCache.txt
done
grep -Ex 'onnxruntime_USE_XNNPACK:(BOOL|UNINITIALIZED)=ON' build/Release/CMakeCache.txt
for option in onnxruntime_ENABLE_WEBASSEMBLY_RELAXED_SIMD onnxruntime_ENABLE_CPU_FP16_OPS XNNPACK_BUILD_ALL_MICROKERNELS; do
  grep -Ex "$option:(BOOL|UNINITIALIZED)=OFF" build/Release/CMakeCache.txt
done

# Audit configured native compilation, not just requested CMake options.
mkdir -p build/smoke
python "$builder/tests/verify-xnnpack-build.py" compile \
  build/Release/compile_commands.json build/smoke/COMPILE_AUDIT.json
# Fail fast on pthreadpool's WASM backend before compiling all of ORT.
cmake --build build/Release --config Release --target pthreadpool --parallel 2
em++ "$builder/tests/wasm-pthreadpool-smoke.cc" \
  -I "$GITHUB_WORKSPACE/dependency-cache/pthreadpool-src/include" \
  "$GITHUB_WORKSPACE/dependency-cache/pthreadpool-build/libpthreadpool.a" \
  -O2 -pthread -msimd128 -fwasm-exceptions -fno-fast-math -ffp-contract=off \
  -Wl,--wrap=pthread_create -sPTHREAD_POOL_SIZE=2 -sSTACK_SIZE=5242880 \
  -sEXIT_RUNTIME=1 -sENVIRONMENT=node \
  -o build/smoke/wasm-pthreadpool-smoke.js
timeout 60 node build/smoke/wasm-pthreadpool-smoke.js 2>&1 | tee build/smoke/PTHREADPOOL_SMOKE.txt

cmake --build build/Release --config Release --target bundling_target --parallel 2
test -s build/Release/libonnxruntime_webassembly.a
python "$builder/tests/make-xnnpack-smoke-model.py" build/smoke/xnnpack-smoke.onnx

# Link only the bundled archive: any omitted XNNPACK dependency must fail here.
# Wrapping pthread_create also checks the actual fallback/XNNPACK worker budget.
em++ "$builder/tests/wasm-xnnpack-smoke.cc" \
  -I include/onnxruntime/core/session \
  build/Release/libonnxruntime_webassembly.a \
  -O2 -pthread -msimd128 -fwasm-exceptions -fno-fast-math -ffp-contract=off \
  -Wl,--wrap=pthread_create \
  -sPTHREAD_POOL_SIZE=4 \
  -sDEFAULT_PTHREAD_STACK_SIZE=2097152 \
  -sSTACK_SIZE=5242880 \
  -sINITIAL_MEMORY=134217728 \
  -sALLOW_MEMORY_GROWTH=1 \
  -sEXIT_RUNTIME=1 \
  -sENVIRONMENT=node \
  --embed-file build/smoke/xnnpack-smoke.onnx@/xnnpack-smoke.onnx \
  -o build/smoke/wasm-xnnpack-smoke.js
timeout 120 node build/smoke/wasm-xnnpack-smoke.js 2>&1 | tee build/smoke/result.txt
python "$builder/tests/verify-xnnpack-build.py" smoke \
  build/smoke/result.txt build/smoke/XNNPACK_SMOKE.json

mkdir -p "$artifact/lib" "$artifact/include"
cp build/Release/libonnxruntime_webassembly.a "$artifact/lib/"
cp -R include/onnxruntime "$artifact/include/"
cp LICENSE README.md ThirdPartyNotices.txt VERSION_NUMBER "$artifact/"
cp "$builder/WASM_THREADED.md" "$builder/xnnpack-static-dependencies.patch" "$artifact/"
cp build/smoke/{COMPILE_AUDIT.json,XNNPACK_SMOKE.json,PTHREADPOOL_SMOKE.txt} "$artifact/"
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
    relaxed_simd: false,
    xnnpack: true,
    pthreadpool_backend: "pthreads",
    pthreadpool_execution_smoke_passed: true,
    xnnpack_source_commit: "fe98e0b93565382648129271381c14d6205255e3",
    pthreadpool_source_commit: "4e80ca24521aa0fb3a746f9ea9c3eaa20e9afbb0",
    fp32_experiment: true,
    fast_math: false,
    fp_contract: "off",
    model_precision_conversion: false,
    source_patches: ["xnnpack-static-dependencies.patch"],
    pthreads: true,
    signed: false,
    exception_abi: "wasm",
    build_flags: ["--build_wasm_static_lib", "--enable_wasm_simd", "--enable_wasm_threads", "--use_xnnpack", "--disable_wasm_exception_catching", "--disable_rtti"],
    required_link_flags: ["-pthread", "-msimd128", "-fwasm-exceptions", "-fno-fast-math", "-ffp-contract=off"],
    thread_pool_scope: "global_ort_plus_per_session_xnnpack",
    smoke_thread_budget: {ort_global_intra_op: 1, ort_global_inter_op: 1, xnnpack_intra_op: 2, ort_worker_pthreads: 0, xnnpack_worker_pthreads: 1},
    smoke_test: {passed: true, runtime: "node", runtime_version: $node_version, execution_provider: "XnnpackExecutionProvider", operators: ["Conv", "ConvTranspose"], dimensions: "1D", profile_verified: true, numerical_reference_verified: true}
  }' > "$artifact/BUILD_INFO.json"
(
  cd "$artifact"
  sha256sum lib/libonnxruntime_webassembly.a xnnpack-static-dependencies.patch COMPILE_AUDIT.json XNNPACK_SMOKE.json PTHREADPOOL_SMOKE.txt > SHA256SUMS
)
