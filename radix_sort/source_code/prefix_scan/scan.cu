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

__device__ __inline__ void scan_4perThread(int thid, int* temp, int* sum)
{
    __shared__ int blocksum[BANK_OFFSET(TOTAL_THREADS / 4)];
    __shared__ int local[BANK_OFFSET(32 * 4)];
    int warpId = thid / 4;
    int a = 4 * thid + 0;
    int b = 4 * thid + 1;
    int c = 4 * thid + 2;
    int d = 4 * thid + 3;
    int e1, e2, e3, e4;
    e1 = temp[BANK_OFFSET(a)];  //= in [t];
    e2 = temp[BANK_OFFSET(b)];  //= in [t + 1];
    e3 = temp[BANK_OFFSET(c)];  //= in [t + 2];
    e4 = temp[BANK_OFFSET(d)];  //= in [t + 3];
    temp[BANK_OFFSET(b)] += temp[BANK_OFFSET(a)];
    temp[BANK_OFFSET(c)] += temp[BANK_OFFSET(b)];
    temp[BANK_OFFSET(d)] += temp[BANK_OFFSET(c)];
    local[BANK_OFFSET(thid)] = temp[BANK_OFFSET(d)];
    __syncthreads();

    if (thid < 32)
    {
        local[BANK_OFFSET(thid * 4 + 1)] += local[BANK_OFFSET(thid * 4 + 0)];
        local[BANK_OFFSET(thid * 4 + 2)] += local[BANK_OFFSET(thid * 4 + 1)];
        local[BANK_OFFSET(thid * 4 + 3)] += local[BANK_OFFSET(thid * 4 + 2)];
    }
    __syncthreads();

    if (thid % 4 != 0)
    {
        int off = thid % 4 - 1;
        int val = local[BANK_OFFSET(warpId * 4 + off)];
        temp[BANK_OFFSET(a)] += val;
        temp[BANK_OFFSET(b)] += val;
        temp[BANK_OFFSET(c)] += val;
        temp[BANK_OFFSET(d)] += val;
    }

    __syncthreads();
    if (thid < 32)
    {
        blocksum[BANK_OFFSET(thid)] = temp[BANK_OFFSET(thid * 16 + 15)];
        if (thid >= 1) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 1)];
        if (thid >= 2) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 2)];
        if (thid >= 4) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 4)];
        if (thid >= 8) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 8)];
        if (thid >= 16) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 16)];
    }
    __syncthreads();

    int add = (warpId > 0) ? blocksum[BANK_OFFSET(warpId - 1)] : 0;
    temp[BANK_OFFSET(a)] = temp[BANK_OFFSET(a)] + add - e1;
    temp[BANK_OFFSET(b)] = temp[BANK_OFFSET(b)] + add - e2;
    temp[BANK_OFFSET(c)] = temp[BANK_OFFSET(c)] + add - e3;
    temp[BANK_OFFSET(d)] = temp[BANK_OFFSET(d)] + add - e4;
    if (thid == TOTAL_THREADS - 1) *sum = temp[BANK_OFFSET(d)] + e4;
    __syncthreads();
}

static inline int nextPow2(int n)
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

__global__ void add_kernel(int* device_result, int* device_blocksum)
{
    __shared__ int temp1;
    int thid = threadIdx.x;
    int N = blockDim.x;
    if (thid == 0) temp1 = device_blocksum[blockIdx.x];
    __syncthreads();
    device_result[blockIdx.x * 4 * blockDim.x + thid] = device_result[blockIdx.x * 4 * blockDim.x + thid] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + N] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] + temp1;
}
__global__ void sweep_kernel(int N, int* device_x, int* device_result, int* device_blocksum)
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

    scan_4perThread(thid, temp, &device_blocksum[blockIdx.x]);
    device_result[offset + a1] = temp[a];
    device_result[offset + b1] = temp[b];
    device_result[offset + c1] = temp[c];
    device_result[offset + d1] = temp[d];
}

int** device_blocksum;
int total_levels;

void alloc_blocksum(int N, int tpb)
{
    int level = 0;
    int m = N;
    do
    {
        m = (m + 4 * tpb - 1) / (4 * tpb);
        level++;
    } while (m > 1);
    device_blocksum = (int**)malloc(level * sizeof(int*));
    total_levels = level;
    level = 0;
    m = N;
    do
    {
        m = (m + 4 * tpb - 1) / (4 * tpb);
        cudaMalloc((void**)&device_blocksum[level], sizeof(int) * m);
        level++;
    } while (level < total_levels);
}
void dealloc_blocksum()
{
    int level = 0;
    for (level = 0; level < total_levels; level++)
    {
        cudaFree(device_blocksum[level]);
    }
    free(device_blocksum);
}

void _exclusive_scan(int* device_start, int length, int* device_result, int level)
{
    int N = nextPow2(length);
    const int threadsPerBlock = TOTAL_THREADS;
    const int blocks = (N + 4 * threadsPerBlock - 1) / (4 * threadsPerBlock);

    if (level == 0)
    {
        sweep_kernel<<<blocks, threadsPerBlock, (BANK_OFFSET(4 * threadsPerBlock)) * sizeof(int)>>>(
            4 * threadsPerBlock, device_start, device_result, device_blocksum[level]);
    }
    else
    {
        sweep_kernel<<<blocks, threadsPerBlock, (BANK_OFFSET(4 * threadsPerBlock)) * sizeof(int)>>>(
            4 * threadsPerBlock, device_start, device_blocksum[level - 1], device_blocksum[level]);
    }
    if (blocks > 1)
    {
        _exclusive_scan(device_start, blocks, device_blocksum[level], level + 1);
        if (level > 0)
        {
            add_kernel<<<blocks, threadsPerBlock>>>(device_blocksum[level - 1], device_blocksum[level]);
        }
        else
        {
            add_kernel<<<blocks, threadsPerBlock>>>(device_result, device_blocksum[level]);
        }
    }
}

void exclusive_scan(int* device_start, int length, int* device_result)
{
    alloc_blocksum(length, TOTAL_THREADS);
    _exclusive_scan(device_start, length, device_result, 0);
    dealloc_blocksum();
}

double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void**)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, end - inarray, device_result);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);
    return overallDuration;
}

double cudaScanThrust(int* inarray, int* end, int* resultarray)
{
    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    // cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}
