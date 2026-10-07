// Final archive integration test, not a performance benchmark.
#include <emscripten/threading.h>
#include <onnxruntime_cxx_api.h>
#include <pthread.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>
#include <unordered_map>

#if !defined(__EMSCRIPTEN_PTHREADS__) || !defined(__wasm_simd128__)
#error This smoke test requires pthreads and ordinary WASM SIMD.
#endif
#ifdef __wasm_relaxed_simd__
#error Relaxed SIMD is outside this experiment.
#endif
#ifdef __FAST_MATH__
#error Fast math is outside this experiment.
#endif

static std::atomic<unsigned> pthread_creates{0};
extern "C" int __real_pthread_create(pthread_t*, const pthread_attr_t*,
                                     void* (*)(void*), void*);
extern "C" int __wrap_pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                                     void* (*entry)(void*), void* arg) {
  int result = __real_pthread_create(thread, attr, entry, arg);
  if (result == 0) ++pthread_creates;
  return result;
}

static std::array<float, 64> reference(const std::array<float, 64>& input) {
  std::array<float, 128> hidden{};
  std::array<float, 64> output{};
  for (int m = 0; m < 8; ++m) {
    for (int w = 0; w < 16; ++w) {
      float sum = (m - 3) / 32.0f;
      for (int c = 0; c < 4; ++c) {
        for (int k = 0; k < 3; ++k) {
          const int x = w + k - 1;
          if (x >= 0 && x < 16)
            sum += input[c * 16 + x] * (((m * 12 + c * 3 + k) % 7 - 3) / 16.0f);
        }
      }
      hidden[m * 16 + w] = sum;
    }
  }
  for (int c = 0; c < 4; ++c) {
    for (int w = 0; w < 16; ++w) output[c * 16 + w] = (c - 1) / 32.0f;
  }
  for (int m = 0; m < 8; ++m) {
    for (int w = 0; w < 16; ++w) {
      for (int c = 0; c < 4; ++c) {
        for (int k = 0; k < 3; ++k) {
          const int x = w + k - 1;
          if (x >= 0 && x < 16)
            output[c * 16 + x] += hidden[m * 16 + w] * (((m * 12 + c * 3 + k) % 5 - 2) / 16.0f);
        }
      }
    }
  }
  return output;
}

static void run_and_check(Ort::Session& session, std::array<float, 64>& input,
                          const std::array<float, 64>& expected) {
  const std::array<int64_t, 3> shape{1, 4, 16};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                               shape.data(), shape.size());
  const char* inputs[] = {"X"};
  const char* outputs[] = {"Y"};
  auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
  if (result.size() != 1 || result[0].GetTensorTypeAndShapeInfo().GetElementCount() != 64)
    throw std::runtime_error("Unexpected output shape");
  const float* actual = result[0].GetTensorData<float>();
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-6f)
      throw std::runtime_error("Conv1d/ConvTranspose1d reference mismatch");
  }
}

int run_smoke() {
  if (!emscripten_has_threading_support()) return 1;
  bool worker_ran = false;
  const auto main_thread = std::this_thread::get_id();
  std::thread worker([&] { worker_ran = std::this_thread::get_id() != main_thread; });
  worker.join();
  if (!worker_ran || pthread_creates != 1) return 2;
  pthread_creates = 0;

  // Threaded WASM uses the global ORT pool. A one-thread global pool prevents
  // fallback nodes from owning another compute pool beside XNNPACK's pool.
  Ort::ThreadingOptions threading;
  threading.SetGlobalIntraOpNumThreads(1);
  threading.SetGlobalInterOpNumThreads(1);
  threading.SetGlobalSpinControl(0);
  Ort::Env env(threading, ORT_LOGGING_LEVEL_WARNING, "wasm-xnnpack-smoke");
  Ort::SessionOptions cpu_options;
  cpu_options.DisablePerSessionThreads();
  cpu_options.SetExecutionMode(ORT_SEQUENTIAL);
  cpu_options.SetIntraOpNumThreads(1);
  cpu_options.SetInterOpNumThreads(1);
  cpu_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  Ort::Session cpu(env, "/xnnpack-smoke.onnx", cpu_options);
  const unsigned cpu_threads = pthread_creates.load();
  if (cpu_threads != 0) throw std::runtime_error("Unexpected ORT fallback worker pool");

  auto options = cpu_options.Clone();
  options.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", "2"}});
  options.EnableProfiling("/xnnpack-profile");
  Ort::Session xnnpack(env, "/xnnpack-smoke.onnx", options);
  const unsigned xnnpack_threads = pthread_creates.load() - cpu_threads;
  if (xnnpack_threads != 1) throw std::runtime_error("Expected one XNNPACK pthread worker");

  std::array<float, 64> input{};
  for (size_t i = 0; i < input.size(); ++i) input[i] = (static_cast<int>(i % 13) - 6) / 8.0f;
  const auto expected = reference(input);
  run_and_check(cpu, input, expected);
  run_and_check(xnnpack, input, expected);
  if (pthread_creates != 1) throw std::runtime_error("Unexpected extra compute workers");

  Ort::AllocatorWithDefaultOptions allocator;
  // ORT's WASM profiler writes JSON to std::cout rather than to MEMFS.
  std::puts("PROFILE_JSON_BEGIN");
  std::fflush(stdout);
  auto profile_path = xnnpack.EndProfilingAllocated(allocator);
  std::fflush(stdout);
  std::puts("PROFILE_JSON_END");
  if (!profile_path || !*profile_path) throw std::runtime_error("ORT profiling was not enabled");
  std::puts("THREAD_COUNTS {\"ort_fallback_workers\":0,\"xnnpack_workers\":1,\"xnnpack_intra_op_threads\":2,\"ort_global_intra_op_threads\":1,\"ort_global_inter_op_threads\":1}");
  std::puts("PASS: pthread create/join; CPU and XNNPACK FP32 Conv1d/ConvTranspose1d values match scalar reference; profile verification required");
  return 0;
}

int main() {
  try { return run_smoke(); }
  catch (const Ort::Exception& error) {
    std::fprintf(stderr, "ORT status %d: %s\n", error.GetOrtErrorCode(), error.what());
    return 90;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Smoke exception: %s\n", error.what());
    return 91;
  }
}
