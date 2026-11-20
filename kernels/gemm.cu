#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_DIM 16

// 2D Shared memory tiled GEMM
extern "C" __global__ void gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int k_a = t * TILE_DIM + tx;
        int k_b = t * TILE_DIM + ty;

        tile_A[ty][tx] = (row < M && k_a < K) ? __half2float(A[row * K + k_a]) : 0.0f;
        tile_B[ty][tx] = (k_b < K && col < N) ? __half2float(B[k_b * N + col]) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += tile_A[ty][k] * tile_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

extern "C" __declspec(dllexport) void launch_gemm(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_kernel<<<grid, block, 0, stream>>>(
        (const half*)A, (const half*)B, (half*)C, M, N, K
    );
}
