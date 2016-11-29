#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "CycleTimer.h"
#define ITER 4
#define BANK_OFFSET1(n) (n) + (((n) >> 5))
#define BANK_OFFSET(n) (n) + (((n) >> 5))
#define NUM_BLOCKS(length, dim) nextPow2(length) / (2 * dim)
#define ELEM 4
#define TOTAL_THREADS 128
#define TWO_PWR(n) (1 << (n))
extern float toBW(int bytes, float sec);

__device__ __inline__ void scan_4elements(int thid, int* temp, int* sum)
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

__device__ __inline__ void prefix_sum_warp(int thid, int* temp, int N)
{
    if (thid < 16)
    {
        int i = temp[thid];
        if (thid >= 1) temp[thid] += temp[thid - 1];
        if (thid >= 2) temp[thid] += temp[thid - 2];
        if (thid >= 4) temp[thid] += temp[thid - 4];
        if (thid >= 8) temp[thid] += temp[thid - 8];
        temp[thid] -= i;
    }
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
    int temp1;
    int thid = threadIdx.x;
    int N = blockDim.x;
    int offset = blockIdx.x * 4 * blockDim.x;

    temp1 = device_blocksum[blockIdx.x];
    device_result[offset + thid] = device_result[offset + thid] + temp1;
    device_result[offset + thid + N] = device_result[offset + thid + N] + temp1;
    device_result[offset + thid + 2 * N] = device_result[offset + thid + 2 * N] + temp1;
    device_result[offset + thid + 3 * N] = device_result[offset + thid + 3 * N] + temp1;
}

__global__ void sweep_kernel(int N, int* device_x, int* device_result, int* device_blocksum)
{
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = blockIdx.x * 4 * blockDim.x;

    int a = thid;
    int b = thid + N / 4;
    int c = thid + 2 * N / 4;
    int d = thid + 3 * N / 4;
    int a1 = thid;
    int b1 = thid + N / 4;
    int c1 = thid + 2 * N / 4;
    int d1 = thid + 3 * N / 4;
    a = BANK_OFFSET(a);
    b = BANK_OFFSET(b);
    c = BANK_OFFSET(c);
    d = BANK_OFFSET(d);
    temp[a] = device_result[offset + a1];
    temp[b] = device_result[offset + b1];
    temp[c] = device_result[offset + c1];
    temp[d] = device_result[offset + d1];
    __syncthreads();

    scan_4elements(thid, temp, &device_blocksum[blockIdx.x]);

    device_result[offset + a1] = temp[a];
    device_result[offset + b1] = temp[b];
    device_result[offset + c1] = temp[c];
    device_result[offset + d1] = temp[d];
}

__device__ __inline__ void hist_calc(int thid, int* sm, int nibble, int a, int b, int c, int d, int* hist, int sum)
{
    __shared__ int bit_vect[BANK_OFFSET(4 * TOTAL_THREADS)];
    __shared__ int bit_vect_p[BANK_OFFSET(4 * TOTAL_THREADS)];
    __shared__ int temp_hist[TWO_PWR(ITER)];
    __shared__ int sum1;
    int nibble1 = nibble << 2;
    int sm_a = (sm[BANK_OFFSET1(a)] >> (nibble1)) & (TWO_PWR(ITER) - 1);
    int sm_b = (sm[BANK_OFFSET1(b)] >> (nibble1)) & (TWO_PWR(ITER) - 1);
    int sm_c = (sm[BANK_OFFSET1(c)] >> (nibble1)) & (TWO_PWR(ITER) - 1);
    int sm_d = (sm[BANK_OFFSET1(d)] >> (nibble1)) & (TWO_PWR(ITER) - 1);
    int sm_d1 = (sm[BANK_OFFSET1(d + 1)] >> (nibble1)) & (TWO_PWR(ITER) - 1);
    bit_vect[BANK_OFFSET(a)] = (sm_a != sm_b);
    bit_vect[BANK_OFFSET(b)] = (sm_b != sm_c);
    bit_vect[BANK_OFFSET(c)] = (sm_c != sm_d);
    bit_vect[BANK_OFFSET(d)] = ((d + 1) < (4 * TOTAL_THREADS)) ? (sm_d != sm_d1) : 1;

    bit_vect_p[BANK_OFFSET(a)] = bit_vect[BANK_OFFSET(a)];
    bit_vect_p[BANK_OFFSET(b)] = bit_vect[BANK_OFFSET(b)];
    bit_vect_p[BANK_OFFSET(c)] = bit_vect[BANK_OFFSET(c)];
    bit_vect_p[BANK_OFFSET(d)] = bit_vect[BANK_OFFSET(d)];

    scan_4elements(thid, bit_vect_p, &sum1);

    if (bit_vect[BANK_OFFSET(a)] == 1) temp_hist[bit_vect_p[BANK_OFFSET(a)]] = a + 1;
    if (bit_vect[BANK_OFFSET(b)] == 1) temp_hist[bit_vect_p[BANK_OFFSET(b)]] = b + 1;
    if (bit_vect[BANK_OFFSET(c)] == 1) temp_hist[bit_vect_p[BANK_OFFSET(c)]] = c + 1;
    if (bit_vect[BANK_OFFSET(d)] == 1) temp_hist[bit_vect_p[BANK_OFFSET(d)]] = d + 1;
    __syncthreads();

    if (thid < 32)
    {
        if (thid > 0 && thid <= sum1)
        {
            temp_hist[thid] -= temp_hist[thid - 1];
        }
    }
    __syncthreads();
    if (bit_vect[BANK_OFFSET(a)] == 1) hist[sm_a] = temp_hist[bit_vect_p[BANK_OFFSET(a)]];
    if (bit_vect[BANK_OFFSET(b)] == 1) hist[sm_b] = temp_hist[bit_vect_p[BANK_OFFSET(b)]];
    if (bit_vect[BANK_OFFSET(c)] == 1) hist[sm_c] = temp_hist[bit_vect_p[BANK_OFFSET(c)]];
    if (bit_vect[BANK_OFFSET(d)] == 1) hist[sm_d] = temp_hist[bit_vect_p[BANK_OFFSET(d)]];
}

__global__ void tile_sort(int nibble, int* device_input, int length, int num_blocks, int* device_hist,
                          int* pdevice_hist)
{
    __shared__ int sm[BANK_OFFSET(4 * TOTAL_THREADS)];
    __shared__ int temp[BANK_OFFSET(4 * TOTAL_THREADS)];
    __shared__ int hist[TWO_PWR(ITER)];
    __shared__ int sum;

    int t = 4 * blockIdx.x * blockDim.x + threadIdx.x;
    int thid = threadIdx.x;
    int a = (4 * thid) + 0;
    int b = (4 * thid) + 1;
    int c = (4 * thid) + 2;
    int d = (4 * thid) + 3;
    int N = TOTAL_THREADS;
    int num_0;

    int a1 = thid;
    int b1 = thid + 1 * N;
    int c1 = thid + 2 * N;
    int d1 = thid + 3 * N;
    if (thid < 4)
    {
        hist[4 * thid] = 0;
        hist[4 * thid + 1] = 0;
        hist[4 * thid + 2] = 0;
        hist[4 * thid + 3] = 0;
    }

    sm[BANK_OFFSET1(a1)] = device_input[t];
    sm[BANK_OFFSET1(b1)] = device_input[t + 1 * N];
    sm[BANK_OFFSET1(c1)] = device_input[t + 2 * N];
    sm[BANK_OFFSET1(d1)] = device_input[t + 3 * N];

    __syncthreads();

#pragma unroll
    for (int i = 0; i < ITER; i++)
    {
        int cur_1 = sm[BANK_OFFSET1(a)];
        int cur_2 = sm[BANK_OFFSET1(b)];
        int cur_3 = sm[BANK_OFFSET1(c)];
        int cur_4 = sm[BANK_OFFSET1(d)];
        int bit1 = temp[BANK_OFFSET(a)] = (cur_1 >> (i + nibble * 4)) & 0x1;
        int bit2 = temp[BANK_OFFSET(b)] = (cur_2 >> (i + nibble * 4)) & 0x1;
        int bit3 = temp[BANK_OFFSET(c)] = (cur_3 >> (i + nibble * 4)) & 0x1;
        int bit4 = temp[BANK_OFFSET(d)] = (cur_4 >> (i + nibble * 4)) & 0x1;

        scan_4elements(thid, temp, &sum);

        num_0 = 4 * TOTAL_THREADS - sum;
        if (bit1)
            sm[BANK_OFFSET1(temp[BANK_OFFSET(a)] + num_0)] = cur_1;
        else
            sm[BANK_OFFSET1(a - temp[BANK_OFFSET(a)])] = cur_1;
        if (bit2)
            sm[BANK_OFFSET1(temp[BANK_OFFSET(b)] + num_0)] = cur_2;
        else
            sm[BANK_OFFSET1(b - temp[BANK_OFFSET(b)])] = cur_2;
        if (bit3)
            sm[BANK_OFFSET1(temp[BANK_OFFSET(c)] + num_0)] = cur_3;
        else
            sm[BANK_OFFSET1(c - temp[BANK_OFFSET(c)])] = cur_3;
        if (bit4)
            sm[BANK_OFFSET1(temp[BANK_OFFSET(d)] + num_0)] = cur_4;
        else
            sm[BANK_OFFSET1(d - temp[BANK_OFFSET(d)])] = cur_4;
        __syncthreads();
    }

    hist_calc(thid, sm, nibble, a, b, c, d, hist, sum);
    device_input[t] = sm[BANK_OFFSET1(a1)];
    device_input[t + 1 * N] = sm[BANK_OFFSET1(b1)];
    device_input[t + 2 * N] = sm[BANK_OFFSET1(c1)];
    device_input[t + 3 * N] = sm[BANK_OFFSET1(d1)];
    __syncthreads();

    if (thid < ITER)
    {
        device_hist[a * num_blocks + blockIdx.x] = hist[a];
        device_hist[(b)*num_blocks + blockIdx.x] = hist[b];
        device_hist[(c)*num_blocks + blockIdx.x] = hist[c];
        device_hist[(d)*num_blocks + blockIdx.x] = hist[d];
        pdevice_hist[a * num_blocks + blockIdx.x] = hist[a];
        pdevice_hist[(b)*num_blocks + blockIdx.x] = hist[b];
        pdevice_hist[(c)*num_blocks + blockIdx.x] = hist[c];
        pdevice_hist[(d)*num_blocks + blockIdx.x] = hist[d];
    }
}

__global__ void output_index(int* device_hist, int* pdevice_hist, int* device_input, int* device_out, int length,
                             int num_blocks, int nibble)
{
    __shared__ int temp[TWO_PWR(ITER)];
    int t = 4 * blockIdx.x * blockDim.x + threadIdx.x;
    int N = TOTAL_THREADS;
    int thid = threadIdx.x;

    if (t < length)
    {
        int val1;
        int val2;
        int val3;
        int val4;
        int nibble1 = nibble << 2;
        int lindex1;
        int lindex2;
        int lindex3;
        int lindex4;
        int gindex1;
        int gindex2;
        int gindex3;
        int gindex4;
        int a = t;
        int b = t + 1 * N;
        int c = t + 2 * N;
        int d = t + 3 * N;
        int a1 = thid;
        int b1 = thid + 1 * N;
        int c1 = thid + 2 * N;
        int d1 = thid + 3 * N;
        val1 = device_input[a];
        val2 = device_input[b];
        val3 = device_input[c];
        val4 = device_input[d];

        if (thid < 32)
        {
            if ((thid) < ITER)
            {
                temp[4 * thid] = device_hist[4 * thid * num_blocks + blockIdx.x];
                temp[4 * thid + 1] = device_hist[(4 * thid + 1) * num_blocks + blockIdx.x];
                temp[4 * thid + 2] = device_hist[(4 * thid + 2) * num_blocks + blockIdx.x];
                temp[4 * thid + 3] = device_hist[(4 * thid + 3) * num_blocks + blockIdx.x];
            }

            prefix_sum_warp(thid, temp, TWO_PWR(ITER));
        }
        __syncthreads();
        lindex1 = temp[((val1 >> (nibble1)) & ((1 << ITER) - 1))];
        lindex2 = temp[((val2 >> (nibble1)) & ((1 << ITER) - 1))];
        lindex3 = temp[((val3 >> (nibble1)) & ((1 << ITER) - 1))];
        lindex4 = temp[((val4 >> (nibble1)) & ((1 << ITER) - 1))];
        gindex1 = pdevice_hist[((val1 >> (nibble1)) & ((1 << ITER) - 1)) * num_blocks + blockIdx.x];
        gindex2 = pdevice_hist[((val2 >> (nibble1)) & ((1 << ITER) - 1)) * num_blocks + blockIdx.x];
        gindex3 = pdevice_hist[((val3 >> (nibble1)) & ((1 << ITER) - 1)) * num_blocks + blockIdx.x];
        gindex4 = pdevice_hist[((val4 >> (nibble1)) & ((1 << ITER) - 1)) * num_blocks + blockIdx.x];

        device_out[a1 + gindex1 - lindex1] = val1;
        device_out[b1 + gindex2 - lindex2] = val2;
        device_out[c1 + gindex3 - lindex3] = val3;
        device_out[d1 + gindex4 - lindex4] = val4;
    }
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

void exclusive_scan(int* device_start, int length, int* device_result, int level)
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
        exclusive_scan(device_start, blocks, device_blocksum[level], level + 1);
        if (level > 0)
        {
            add_kernel<<<blocks, threadsPerBlock, sizeof(int)>>>(device_blocksum[level - 1], device_blocksum[level]);
        }
        else
        {
            add_kernel<<<blocks, threadsPerBlock, sizeof(int)>>>(device_result, device_blocksum[level]);
        }
    }
}

void radix_sort(int* device_start, int length, int* device_result, int num_blocks, int* device_hist, int* pdevice_hist)
{
    for (int nibble = 0; nibble < 8; nibble++)
    {
        if (nibble % 2 == 0)
        {
            tile_sort<<<num_blocks, TOTAL_THREADS>>>(nibble, device_start, length, num_blocks, device_hist,
                                                     pdevice_hist);

            //   thrust::exclusive_scan(device_hist, device_hist + num_blocks*16, pdevice_hist);
            exclusive_scan(pdevice_hist, num_blocks * TWO_PWR(ITER), pdevice_hist, 0);

            output_index<<<num_blocks, TOTAL_THREADS>>>(device_hist, pdevice_hist, device_start, device_result, length,
                                                        num_blocks, nibble);
        }
        else
        {
            tile_sort<<<num_blocks, TOTAL_THREADS>>>(nibble, device_result, length, num_blocks, device_hist,
                                                     pdevice_hist);

            //   thrust::exclusive_scan(device_hist, device_hist + num_blocks*16, pdevice_hist);
            exclusive_scan(pdevice_hist, num_blocks * TWO_PWR(ITER), pdevice_hist, 0);

            output_index<<<num_blocks, TOTAL_THREADS>>>(device_hist, pdevice_hist, device_result, device_start, length,
                                                        num_blocks, nibble);
        }
    }
}

double cudaSort(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int rounded_length = 512 * (1 + (end - inarray - 1) / 512);
    int *device_hist, *pdevice_hist;
    int num_blocks = (rounded_length + 4 * TOTAL_THREADS - 1) / (4 * TOTAL_THREADS);
    int len = num_blocks * TWO_PWR(ITER);

    int temp[rounded_length - (end - inarray)];
    for (int i = 0; i < rounded_length - (end - inarray); i++)
    {
        temp[i] = INT_MAX;
    }

    cudaMalloc((void**)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_input, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_hist, sizeof(int) * num_blocks * TWO_PWR(ITER));
    cudaMalloc((void**)&pdevice_hist, sizeof(int) * num_blocks * TWO_PWR(ITER));

    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();
    cudaMemcpy(device_input + (end - inarray), temp, (rounded_length - (end - inarray)) * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();
    cudaMemcpy(device_result + (end - inarray), temp, (rounded_length - (end - inarray)) * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    alloc_blocksum(nextPow2(len), TOTAL_THREADS);
    double startTime = CycleTimer::currentSeconds();
    radix_sort(device_input, rounded_length, device_result, num_blocks, device_hist, pdevice_hist);
    cudaThreadSynchronize();
    // Wait for any work left over to be completed.
    double endTime = CycleTimer::currentSeconds();
    float overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);
    dealloc_blocksum();
    cudaFree(device_hist);
    cudaFree(pdevice_hist);
    return overallDuration;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
