// Check the real pinned pthreadpool backend, independently of ORT/XNNPACK.
#include <pthreadpool.h>
#include <pthread.h>
#include <array>
#include <atomic>
#include <cstdio>

static std::atomic<unsigned> creates{0};
extern "C" int __real_pthread_create(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
extern "C" int __wrap_pthread_create(pthread_t* t, const pthread_attr_t* a, void* (*f)(void*), void* p) {
  const int rc = __real_pthread_create(t, a, f, p);
  if (rc == 0) ++creates;
  return rc;
}

struct Context {
  pthread_t caller = pthread_self();
  std::atomic<unsigned> thread_mask{0};
  std::array<unsigned, 8192> values{};
};

static void task(void* raw, size_t i) {
  auto& context = *static_cast<Context*>(raw);
  context.thread_mask.fetch_or(pthread_equal(context.caller, pthread_self()) ? 1u : 2u);
  volatile unsigned value = static_cast<unsigned>(i);
  for (unsigned k = 0; k < 1000; ++k) value = value + k;
  context.values[i] = value;
}

int main() {
  auto pool = pthreadpool_create(2);
  if (!pool || pthreadpool_get_threads_count(pool) != 2 || creates != 1) return 1;
  Context context;
  pthreadpool_parallelize_1d(pool, task, &context, context.values.size(), 0);
  for (size_t i = 0; i < context.values.size(); ++i) {
    if (context.values[i] != i + 499500) return 2;
  }
  if (context.thread_mask != 3) return 3;
  pthreadpool_destroy(pool);
  std::puts("PASS: pinned pthreadpool threads=2, one worker pthread, caller+worker executed tasks, all results verified");
  return 0;
}
