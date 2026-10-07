// Fast final-link test of XNNPACK itself, before the much larger ORT build.
#include <xnnpack.h>
#include <pthreadpool.h>
#include <array>
#include <vector>
#include <limits>
#include <cstdio>
#include <cstdint>
#if !defined(__EMSCRIPTEN_PTHREADS__) || !defined(__wasm_simd128__) || defined(__wasm_relaxed_simd__) || defined(__FAST_MATH__)
#error This preflight requires strict floating point, ordinary SIMD and pthreads.
#endif
#define REQUIRE_OK(x) do { auto st=(x); if(st != xnn_status_success) { std::fprintf(stderr,"status %d at line %d\n",st,__LINE__); return 1; } } while(0)
int main() {
 REQUIRE_OK(xnn_initialize(nullptr));
 auto pool=pthreadpool_create(2); if(!pool || pthreadpool_get_threads_count(pool)!=2)return 2;
 std::array<float,64+16> input; input.fill(1.0f);
 std::array<float,96> kernel; kernel.fill(0.25f);
 std::array<float,8> bias{};
 std::array<float,128+16> output{};
 xnn_operator_t op=nullptr;
 REQUIRE_OK(xnn_create_convolution2d_nhwc_f32(0,1,0,1,1,3,1,1,1,1,1,4,8,4,8,kernel.data(),bias.data(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),0,nullptr,nullptr,&op));
 size_t size=0,align=0,height=0,width=0;
 REQUIRE_OK(xnn_reshape_convolution2d_nhwc_f32(op,1,1,16,&size,&align,&height,&width,pool));
 if(height!=1 || width!=16 || align==0)return 3;
 std::vector<uint8_t> allocation(size+align);
 void* workspace=reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(allocation.data())+align-1)&~(align-1));
 REQUIRE_OK(xnn_setup_convolution2d_nhwc_f32(op,workspace,input.data(),output.data()));
 REQUIRE_OK(xnn_run_operator(op,pool));
 for(size_t w=0;w<16;++w)for(size_t c=0;c<8;++c)if(output[w*8+c]!=(w==0||w==15?2.0f:3.0f))return 4;
 REQUIRE_OK(xnn_delete_operator(op));pthreadpool_destroy(pool);REQUIRE_OK(xnn_deinitialize());
 std::puts("PASS: pinned XNNPACK ordinary-SIMD strict-FP32 1D-shaped convolution with two-thread pool");
}
