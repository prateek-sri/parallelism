#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "CycleTimer.h"
#define BANK_OFFSET(n) (n)
#define TOTAL_THREADS 128
#define ELEM 4

__device__ __inline__ void scan_loopUnroll(int thid, int N, int* temp, int* device_blocksum)
{
    int t;

    int a1;
    int b1;
    int c1;
    int d1;
    a1 = ((thid << 2) + 1) - 1;
    b1 = ((thid << 2) + 2) - 1;
    c1 = ((thid << 2) + 3) - 1;
    d1 = ((thid << 2) + 4) - 1;
    a1 = BANK_OFFSET(a1);
    b1 = BANK_OFFSET(b1);
    c1 = BANK_OFFSET(c1);
    d1 = BANK_OFFSET(d1);
    temp[b1] += temp[a1];
    temp[d1] += temp[c1];
    __syncthreads();
    if (thid<TOTAL_THREADS>> 1)  // 256
    {
        a1 = (((thid << 2) + 1) << 1) - 1;
        b1 = (((thid << 2) + 2) << 1) - 1;
        c1 = (((thid << 2) + 3) << 1) - 1;
        d1 = (((thid << 2) + 4) << 1) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        temp[b1] += temp[a1];
        temp[d1] += temp[c1];
    }
    __syncthreads();
    if (thid<TOTAL_THREADS>> 2)  // 128
    {
        a1 = (((thid << 2) + 1) << 2) - 1;
        b1 = (((thid << 2) + 2) << 2) - 1;
        c1 = (((thid << 2) + 3) << 2) - 1;
        d1 = (((thid << 2) + 4) << 2) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        temp[b1] += temp[a1];
        temp[d1] += temp[c1];
    }
    if (thid<TOTAL_THREADS>> 3)  // 64
    {
        a1 = (((thid << 2) + 1) << 3) - 1;
        b1 = (((thid << 2) + 2) << 3) - 1;
        c1 = (((thid << 2) + 3) << 3) - 1;
        d1 = (((thid << 2) + 4) << 3) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        temp[b1] += temp[a1];
        temp[d1] += temp[c1];
    }
    if (thid<TOTAL_THREADS>> 4)  // 32
    {
        a1 = (((thid << 2) + 1) << 4) - 1;
        b1 = (((thid << 2) + 2) << 4) - 1;
        c1 = (((thid << 2) + 3) << 4) - 1;
        d1 = (((thid << 2) + 4) << 4) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        temp[b1] += temp[a1];
        temp[d1] += temp[c1];
    }
    if (thid<TOTAL_THREADS>> 5)  // 16
    {
        a1 = (((thid << 2) + 1) << 5) - 1;
        b1 = (((thid << 2) + 2) << 5) - 1;
        c1 = (((thid << 2) + 3) << 5) - 1;
        d1 = (((thid << 2) + 4) << 5) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        temp[b1] += temp[a1];
        temp[d1] += temp[c1];
    }
    if (thid<TOTAL_THREADS>> 6)  // 8
    {
        a1 = (((thid << 2) + 1) << 6) - 1;
        b1 = (((thid << 2) + 2) << 6) - 1;
        c1 = (((thid << 2) + 3) << 6) - 1;
        d1 = (((thid << 2) + 4) << 6) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        temp[b1] += temp[a1];
        temp[d1] += temp[c1];
    }
    if (thid<TOTAL_THREADS>> 7)  // 4
    {
        a1 = (((thid << 2) + 1) << 7) - 1;
        b1 = (((thid << 2) + 2) << 7) - 1;
        c1 = (((thid << 2) + 3) << 7) - 1;
        d1 = (((thid << 2) + 4) << 7) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        temp[b1] += temp[a1];
        temp[d1] += temp[c1];
    }
    if (thid == 0)
    {
        a1 = 4 * TOTAL_THREADS / 2 - 1;
        b1 = 4 * TOTAL_THREADS - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
        *device_blocksum = temp[BANK_OFFSET(N - 1)];
        temp[BANK_OFFSET(N - 1)] = 0;
        temp[b1] = temp[a1];
        temp[a1] = 0;
    }
    if (thid<TOTAL_THREADS>> 7)  // 4
    {
        a1 = (((thid << 2) + 1) << 7) - 1;
        b1 = (((thid << 2) + 2) << 7) - 1;
        c1 = (((thid << 2) + 3) << 7) - 1;
        d1 = (((thid << 2) + 4) << 7) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
        t = temp[c1];
        temp[c1] = temp[d1];
        temp[d1] += t;
    }
    if (thid<TOTAL_THREADS>> 6)  // 8
    {
        a1 = (((thid << 2) + 1) << 6) - 1;
        b1 = (((thid << 2) + 2) << 6) - 1;
        c1 = (((thid << 2) + 3) << 6) - 1;
        d1 = (((thid << 2) + 4) << 6) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
        t = temp[c1];
        temp[c1] = temp[d1];
        temp[d1] += t;
    }
    if (thid<TOTAL_THREADS>> 5)  // 16
    {
        a1 = (((thid << 2) + 1) << 5) - 1;
        b1 = (((thid << 2) + 2) << 5) - 1;
        c1 = (((thid << 2) + 3) << 5) - 1;
        d1 = (((thid << 2) + 4) << 5) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
        t = temp[c1];
        temp[c1] = temp[d1];
        temp[d1] += t;
    }
    if (thid<TOTAL_THREADS>> 4)  // 32
    {
        a1 = (((thid << 2) + 1) << 4) - 1;
        b1 = (((thid << 2) + 2) << 4) - 1;
        c1 = (((thid << 2) + 3) << 4) - 1;
        d1 = (((thid << 2) + 4) << 4) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
        t = temp[c1];
        temp[c1] = temp[d1];
        temp[d1] += t;
    }
    if (thid<TOTAL_THREADS>> 3)  // 64
    {
        a1 = (((thid << 2) + 1) << 3) - 1;
        b1 = (((thid << 2) + 2) << 3) - 1;
        c1 = (((thid << 2) + 3) << 3) - 1;
        d1 = (((thid << 2) + 4) << 3) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
        t = temp[c1];
        temp[c1] = temp[d1];
        temp[d1] += t;
    }
    __syncthreads();
    if (thid<TOTAL_THREADS>> 2)  // 128
    {
        a1 = (((thid << 2) + 1) << 2) - 1;
        b1 = (((thid << 2) + 2) << 2) - 1;
        c1 = (((thid << 2) + 3) << 2) - 1;
        d1 = (((thid << 2) + 4) << 2) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
        t = temp[c1];
        temp[c1] = temp[d1];
        temp[d1] += t;
    }
    __syncthreads();
    if (thid<TOTAL_THREADS>> 1)  // 256
    {
        a1 = (((thid << 2) + 1) << 1) - 1;
        b1 = (((thid << 2) + 2) << 1) - 1;
        c1 = (((thid << 2) + 3) << 1) - 1;
        d1 = (((thid << 2) + 4) << 1) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        c1 = BANK_OFFSET(c1);
        d1 = BANK_OFFSET(d1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
        t = temp[c1];
        temp[c1] = temp[d1];
        temp[d1] += t;
    }
    __syncthreads();
    a1 = (((thid << 2) + 1)) - 1;
    b1 = (((thid << 2) + 2)) - 1;
    c1 = (((thid << 2) + 3)) - 1;
    d1 = (((thid << 2) + 4)) - 1;
    a1 = BANK_OFFSET(a1);
    b1 = BANK_OFFSET(b1);
    c1 = BANK_OFFSET(c1);
    d1 = BANK_OFFSET(d1);
    t = temp[a1];
    temp[a1] = temp[b1];
    temp[b1] += t;
    t = temp[c1];
    temp[c1] = temp[d1];
    temp[d1] += t;
    __syncthreads();
}

static inline int nextPow2_loopUnroll(int n)
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

__global__ void add_kernel_loopUnroll(int* device_result, int* device_blocksum_loopUnroll)
{
    __shared__ int temp1;
    int thid = threadIdx.x;
    int N = blockDim.x;
    if (thid == 0) temp1 = device_blocksum_loopUnroll[blockIdx.x];
    __syncthreads();
    device_result[blockIdx.x * 4 * blockDim.x + thid] = device_result[blockIdx.x * 4 * blockDim.x + thid] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + N] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] + temp1;
}

__global__ void sweep_kernel_loopUnroll(int N, int* device_x, int* device_result, int* device_blocksum)
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

    scan_loopUnroll(thid, N, temp, &device_blocksum[blockIdx.x]);

    device_result[offset + a1] = temp[a];
    device_result[offset + b1] = temp[b];
    device_result[offset + c1] = temp[c];
    device_result[offset + d1] = temp[d];
}

int** device_blocksum_loopUnroll;
int total_levels_loopUnroll;

void alloc_blocksum_loopUnroll(int N, int tpb)
{
    int level = 0;
    int m = N;
    do
    {
        m = (m + 4 * tpb - 1) / (4 * tpb);
        level++;
    } while (m > 1);
    device_blocksum_loopUnroll = (int**)malloc(level * sizeof(int*));
    total_levels_loopUnroll = level;
    level = 0;
    m = N;
    do
    {
        m = (m + 4 * tpb - 1) / (4 * tpb);
        cudaMalloc((void**)&device_blocksum_loopUnroll[level], sizeof(int) * m);
        level++;
    } while (level < total_levels_loopUnroll);
}
void dealloc_blocksum_loopUnroll()
{
    int level = 0;
    for (level = 0; level < total_levels_loopUnroll; level++)
    {
        cudaFree(device_blocksum_loopUnroll[level]);
    }
    free(device_blocksum_loopUnroll);
}

void _exclusive_scan_loopUnroll(int* device_start, int length, int* device_result, int level)
{
    int N = nextPow2_loopUnroll(length);
    const int threadsPerBlock = TOTAL_THREADS;
    const int blocks = (N + 4 * threadsPerBlock - 1) / (4 * threadsPerBlock);

    if (level == 0)
    {
        sweep_kernel_loopUnroll<<<blocks, threadsPerBlock, (BANK_OFFSET(4 * threadsPerBlock)) * sizeof(int)>>>(
            4 * threadsPerBlock, device_start, device_result, device_blocksum_loopUnroll[level]);
    }
    else
    {
        sweep_kernel_loopUnroll<<<blocks, threadsPerBlock, (BANK_OFFSET(4 * threadsPerBlock)) * sizeof(int)>>>(
            4 * threadsPerBlock, device_start, device_blocksum_loopUnroll[level - 1],
            device_blocksum_loopUnroll[level]);
    }
    if (blocks > 1)
    {
        _exclusive_scan_loopUnroll(device_start, blocks, device_blocksum_loopUnroll[level], level + 1);
        if (level > 0)
        {
            add_kernel_loopUnroll<<<blocks, threadsPerBlock>>>(device_blocksum_loopUnroll[level - 1],
                                                               device_blocksum_loopUnroll[level]);
        }
        else
        {
            add_kernel_loopUnroll<<<blocks, threadsPerBlock>>>(device_result, device_blocksum_loopUnroll[level]);
        }
    }
}

void exclusive_scan_loopUnroll(int* device_start, int length, int* device_result)
{
    alloc_blocksum_loopUnroll(length, TOTAL_THREADS);
    _exclusive_scan_loopUnroll(device_start, length, device_result, 0);
    dealloc_blocksum_loopUnroll();
}

double cudaScan_loopUnroll(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int rounded_length = nextPow2_loopUnroll(end - inarray);
    cudaMalloc((void**)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan_loopUnroll(device_input, end - inarray, device_result);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);
    return overallDuration;
}
