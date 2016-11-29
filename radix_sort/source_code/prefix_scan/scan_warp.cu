#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "CycleTimer.h"
#define BANK_OFFSET(n) (n) + (((n) >> 5))
#define TOTAL_THREADS 128
#define ELEM 4

__device__ __inline__ void prefix32(volatile int* blocksum, int thid2, int off, int* sum)
{
    int thid = thid2 + off;
    int thid1 = thid2 % 32;
    if (thid1 >= 1) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 1)];
    if (thid1 >= 2) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 2)];
    if (thid1 >= 4) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 4)];
    if (thid1 >= 8) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 8)];
    if (thid1 >= 16) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 16)];
    if (thid1 == 31)
    {
        *sum = blocksum[BANK_OFFSET(thid)];
    }
    if (thid1 > 0)
    {
        blocksum[BANK_OFFSET(thid)] = blocksum[BANK_OFFSET(thid - 1)];
    }
    if (thid1 == 0)
    {
        blocksum[BANK_OFFSET(thid)] = 0;
    }
}

__device__ __inline__ void prefix16(volatile int* blocksum, int thid, int off, volatile int* sum)
{
    if (thid >= 1) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 1)];
    if (thid >= 2) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 2)];
    if (thid >= 4) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 4)];
    if (thid >= 8) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 8)];
    if (thid == 15)
    {
        *sum = blocksum[BANK_OFFSET(thid)];
    }
    if (thid > 0)
    {
        blocksum[BANK_OFFSET(thid)] = blocksum[BANK_OFFSET(thid - 1)];
    }
    if (thid == 0)
    {
        blocksum[BANK_OFFSET(thid)] = 0;
    }
}

__device__ __inline__ void scan_warp(int thid, int* temp, int* sum1)
{
    __shared__ int blocksum[BANK_OFFSET(ELEM * 4)];
    int a = BANK_OFFSET(thid);
    int b = BANK_OFFSET(thid + TOTAL_THREADS);
    int c = BANK_OFFSET(thid + TOTAL_THREADS * 2);
    int d = BANK_OFFSET(thid + TOTAL_THREADS * 3);
    int warpId = thid / 32;
    for (int j = 0; j < ELEM; j++)
    {
        prefix32(temp, thid, j * TOTAL_THREADS, &blocksum[BANK_OFFSET(j * 4 + warpId)]);
    }
    if (thid < 4 * ELEM) prefix16(blocksum, thid, 0, sum1);
    __syncthreads();
    temp[a] = temp[a] + blocksum[BANK_OFFSET(warpId)];
    temp[b] = temp[b] + blocksum[BANK_OFFSET(4 + warpId)];   //-e2;
    temp[c] = temp[c] + blocksum[BANK_OFFSET(8 + warpId)];   //-e3;
    temp[d] = temp[d] + blocksum[BANK_OFFSET(12 + warpId)];  // -e4;
    __syncthreads();
}

static inline int nextPow2_warp(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void add_kernel_warp(int* device_result, int* device_blocksum_warp)
{
    __shared__ int temp1;
    int thid = threadIdx.x;
    int N = blockDim.x;
    if (thid == 0) temp1 = device_blocksum_warp[blockIdx.x];
    __syncthreads();
    device_result[blockIdx.x * 4 * blockDim.x + thid] = device_result[blockIdx.x * 4 * blockDim.x + thid] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + N] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] + temp1;
}
__global__ void sweep_kernel_warp(int N, int* device_x, int* device_result, int* device_blocksum_warp)
{
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = blockIdx.x * 4 * blockDim.x;

    int a1 = thid;
    int b1 = thid + N / 4;
    int c1 = thid + 2 * N / 4;
    int d1 = thid + 3 * N / 4;
    int a = BANK_OFFSET(a1);
    int b = BANK_OFFSET(b1);
    int c = BANK_OFFSET(c1);
    int d = BANK_OFFSET(d1);
    temp[a] = device_result[offset + a1];
    temp[b] = device_result[offset + b1];
    temp[c] = device_result[offset + c1];
    temp[d] = device_result[offset + d1];
    __syncthreads();

    scan_warp(thid, temp, &device_blocksum_warp[blockIdx.x]);
    device_result[offset + a1] = temp[a];
    device_result[offset + b1] = temp[b];
    device_result[offset + c1] = temp[c];
    device_result[offset + d1] = temp[d];
}

int** device_blocksum_warp;
int total_levels_warp;

void alloc_blocksum_warp(int N, int tpb)
{
    int level = 0;
    int m = N;
    do
    {
        m = (m + 4 * tpb - 1) / (4 * tpb);
        level++;
    } while (m > 1);
    device_blocksum_warp = (int**)malloc(level * sizeof(int*));
    total_levels_warp = level;
    level = 0;
    m = N;
    do
    {
        m = (m + 4 * tpb - 1) / (4 * tpb);
        cudaMalloc((void**)&device_blocksum_warp[level], sizeof(int) * m);
        level++;
    } while (level < total_levels_warp);
}
void dealloc_blocksum_warp()
{
    int level = 0;
    for (level = 0; level < total_levels_warp; level++)
    {
        cudaFree(device_blocksum_warp[level]);
    }
    free(device_blocksum_warp);
}

void _exclusive_scan_warp(int* device_start, int length, int* device_result, int level)
{
    int N = nextPow2_warp(length);
    const int threadsPerBlock = TOTAL_THREADS;
    const int blocks = (N + 4 * threadsPerBlock - 1) / (4 * threadsPerBlock);

    if (level == 0)
    {
        sweep_kernel_warp<<<blocks, threadsPerBlock, (BANK_OFFSET(4 * threadsPerBlock)) * sizeof(int)>>>(
            4 * threadsPerBlock, device_start, device_result, device_blocksum_warp[level]);
    }
    else
    {
        sweep_kernel_warp<<<blocks, threadsPerBlock, (BANK_OFFSET(4 * threadsPerBlock)) * sizeof(int)>>>(
            4 * threadsPerBlock, device_start, device_blocksum_warp[level - 1], device_blocksum_warp[level]);
    }
    if (blocks > 1)
    {
        _exclusive_scan_warp(device_start, blocks, device_blocksum_warp[level], level + 1);
        if (level > 0)
        {
            add_kernel_warp<<<blocks, threadsPerBlock>>>(device_blocksum_warp[level - 1], device_blocksum_warp[level]);
        }
        else
        {
            add_kernel_warp<<<blocks, threadsPerBlock>>>(device_result, device_blocksum_warp[level]);
        }
    }
}

void exclusive_scan_warp(int* device_start, int length, int* device_result)
{
    alloc_blocksum_warp(length, TOTAL_THREADS);
    _exclusive_scan_warp(device_start, length, device_result, 0);
    dealloc_blocksum_warp();
}

double cudaScan_warp(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int rounded_length = nextPow2_warp(end - inarray);
    cudaMalloc((void**)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan_warp(device_input, end - inarray, device_result);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);
    return overallDuration;
}
