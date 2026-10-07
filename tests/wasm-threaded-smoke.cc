// Smoke test for the final linked archive, not a performance benchmark.
#include <emscripten/threading.h>
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdio>
#include <exception>
#include <thread>

#ifndef __EMSCRIPTEN_PTHREADS__
#error This test must be compiled and linked with -pthread.
#endif
#ifndef __wasm_simd128__
#error This test must be compiled and linked with -msimd128.
#endif

int run_smoke() {
  if (!emscripten_has_threading_support()) {
    std::fputs("Shared-memory threading is unavailable\n", stderr);
    return 1;
  }
  bool worker_ran = false;
  const auto main_thread = std::this_thread::get_id();
  std::thread worker([&] { worker_ran = std::this_thread::get_id() != main_thread; });
  worker.join();  // Synchronizes the worker's write before this read.
  if (!worker_ran) return 2;

  // Threaded WASM defaults to shared environment pools, unlike native/ST builds.
  // Set the real pool size on the environment, not on each session.
  std::fputs("Smoke stage: create global two-thread ORT environment\n", stderr);
  Ort::ThreadingOptions threading;
  threading.SetGlobalIntraOpNumThreads(2);
  threading.SetGlobalInterOpNumThreads(1);
  Ort::Env env(threading, ORT_LOGGING_LEVEL_WARNING, "wasm-threaded-smoke");
  Ort::SessionOptions options;
  options.DisablePerSessionThreads();
  options.SetExecutionMode(ORT_SEQUENTIAL);
  std::fputs("Smoke stage: create two-thread ORT session\n", stderr);
  Ort::Session session(env, "/mul_1.onnx", options);

  // Microsoft's pinned mul_1.onnx multiplies X by [[1,2],[3,4],[5,6]].
  std::array<float, 6> input = {1, 2, 3, 4, 5, 6};
  const std::array<int64_t, 2> shape = {3, 2};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(
      memory, input.data(), input.size(), shape.data(), shape.size());
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  std::fputs("Smoke stage: run inference\n", stderr);
  auto output = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                            output_names, 1);
  if (output.size() != 1 ||
      output.front().GetTensorTypeAndShapeInfo().GetElementCount() != input.size()) {
    return 3;
  }
  const float* actual = output.front().GetTensorData<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    if (actual[i] != input[i] * input[i]) return 4;
  }
  std::puts("PASS: pthread create/join; ORT session intra_op_threads=2; inference values verified");
  return 0;
}

int main() {
  try {
    return run_smoke();
  } catch (const Ort::Exception& error) {
    std::fprintf(stderr, "ORT status %d: %s\n", error.GetOrtErrorCode(), error.what());
    return 90;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Smoke test exception: %s\n", error.what());
    return 91;
  } catch (...) {
    std::fputs("Smoke test raised an unknown exception\n", stderr);
    return 92;
  }
}
