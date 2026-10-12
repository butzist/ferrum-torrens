// Minimal C++ bridge for the two torch CUDA facilities the graph capture needs
// and `tch` 0.22 does not expose: putting the torch runtime's current stream on
// a stream this crate created through the driver API, and routing the
// allocations made while capturing into a private graph mempool so no
// `cudaMalloc` happens between begin and end capture.
//
// Compiled by `build.rs` only under the `cuda` feature, against the libtorch
// headers the devshell exports. It is build-time only: nothing here runs at
// runtime, and it is linked into the same binary as everything else.

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>

#include <utility>

extern "C" {

// Make `raw_stream` the torch current stream for `device`. The stream must have
// been created on the same primary context torch uses.
void ft_set_current_stream(void* raw_stream, int device) {
    auto stream = c10::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(raw_stream),
        static_cast<c10::DeviceIndex>(device));
    c10::cuda::setCurrentCUDAStream(stream);
}

// The private mempool a capture allocates into. `begin` opens the pool and
// starts routing this stream's allocations into it; the handle owns the pool id
// and releasing it stops the routing and frees the pool.
struct FtGraphPool {
    c10::cuda::MempoolId_t id;
    c10::DeviceIndex device;
};

FtGraphPool* ft_pool_begin(void* raw_stream, int device) {
    auto* pool = new FtGraphPool;
    pool->device = static_cast<c10::DeviceIndex>(device);
    pool->id = at::cuda::graph_pool_handle();
    auto raw = static_cast<cudaStream_t>(raw_stream);
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        pool->device, pool->id, [raw](cudaStream_t stream) { return stream == raw; });
    return pool;
}

void ft_pool_end(FtGraphPool* pool) {
    c10::cuda::CUDACachingAllocator::endAllocateToPool(pool->device, pool->id);
    c10::cuda::CUDACachingAllocator::releasePool(pool->device, pool->id);
    delete pool;
}

} // extern "C"
