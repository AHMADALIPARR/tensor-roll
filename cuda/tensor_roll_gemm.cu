// Tensor Roll — Recursive CUDA-Q Model Quantizer
// Copyright (C) 2026 SnapKitty Collective
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Genuine CUDA GEMM kernel: the classical baseline that TensorRoll's
// dispatch boundary (tensor_roll/core.py::dispatch_matmul) would call on
// real NVIDIA hardware.
//
//   C = A x B   (classical region)
//   TensorRoll(A, B) routes candidate blocks to the CUDA-Q kernel family
//   (cudaq/tensor_roll_kernels.py) instead.
//
// Build (requires the NVIDIA CUDA toolkit + a CUDA GPU):
//   nvcc -O3 -arch=sm_80 -o tensor_roll_gemm tensor_roll_gemm.cu
// Run:
//   ./tensor_roll_gemm <M> <N> <K>
//
// NOT compiled or executed in this repository's test environment (no GPU,
// no nvcc). tensor-roll reports cpu-numpy for GEMM here; on CUDA hardware
// this kernel is the drop-in classical region.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE 16

__global__ void tensor_roll_gemm(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        if (row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && t * TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = acc;
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int N = argc > 2 ? atoi(argv[2]) : 512;
    int K = argc > 3 ? atoi(argv[3]) : 512;
    size_t sA = (size_t)M * K * sizeof(float);
    size_t sB = (size_t)K * N * sizeof(float);
    size_t sC = (size_t)M * N * sizeof(float);
    float *hA = (float*)malloc(sA), *hB = (float*)malloc(sB),
          *hC = (float*)malloc(sC);
    for (int i = 0; i < M * K; ++i) hA[i] = (float)(i % 13) / 13.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(i % 7) / 7.0f;
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sA); cudaMalloc(&dB, sB); cudaMalloc(&dC, sC);
    cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    tensor_roll_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost);
    // checksum against a naive host reference on a 16x16 corner
    double ref = 0.0;
    for (int i = 0; i < 16 && i < M; ++i)
        for (int j = 0; j < 16 && j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += (double)hA[i * K + k] * hB[k * N + j];
            ref += fabs(s - hC[i * N + j]);
        }
    double gflops = 2.0 * M * N * K / (ms * 1e6);
    printf("tensor_roll_gemm %dx%dx%d: %.3f ms, %.1f GFLOPS, corner_err=%.2e\n",
           M, N, K, ms, gflops, ref);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
