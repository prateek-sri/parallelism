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
#define BANK_OFFSET(n) (n) + ((n) >> 5)
#define BANK_OFFSET1(n) (n) + ((n) >> 5)
#define NUM_BLOCKS(length, dim) nextPow2_2elements(length) / (2 * dim)
#define TOTAL_THREADS 128
#define TWO_PWR(n) (1 << (n))
extern float toBW(int bytes, float sec);

__device__ __inline__ void scan_4bit(int thid, int* temp, int* sum)
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

    if (thid % 4 == 0)
    {
        local[BANK_OFFSET(warpId * 4 + 1)] += local[BANK_OFFSET(warpId * 4 + 0)];
        local[BANK_OFFSET(warpId * 4 + 2)] += local[BANK_OFFSET(warpId * 4 + 1)];
        local[BANK_OFFSET(warpId * 4 + 3)] += local[BANK_OFFSET(warpId * 4 + 2)];
    }
    __syncthreads();

    if (thid % 4 != 0)
    {
        int off = thid % 4 - 1;
        temp[BANK_OFFSET(a)] += local[BANK_OFFSET(warpId * 4 + off)];
        temp[BANK_OFFSET(b)] += local[BANK_OFFSET(warpId * 4 + off)];
        temp[BANK_OFFSET(c)] += local[BANK_OFFSET(warpId * 4 + off)];
        temp[BANK_OFFSET(d)] += local[BANK_OFFSET(warpId * 4 + off)];
        if (thid % 3 == 0)
        {
            blocksum[BANK_OFFSET(warpId)] = temp[BANK_OFFSET(warpId * 16 + 15)];
        }
    }
    __syncthreads();

    if (thid < 32)
    {
        if (thid >= 1) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 1)];
        if (thid >= 2) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 2)];
        if (thid >= 4) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 4)];
        if (thid >= 8) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 8)];
        if (thid >= 16) blocksum[BANK_OFFSET(thid)] += blocksum[BANK_OFFSET(thid - 16)];
    }
    __syncthreads();

    int add = 0;
    if (warpId > 0) add = blocksum[BANK_OFFSET(warpId - 1)];
    temp[BANK_OFFSET(a)] = temp[BANK_OFFSET(a)] + add - e1;
    temp[BANK_OFFSET(b)] = temp[BANK_OFFSET(b)] + add - e2;
    temp[BANK_OFFSET(c)] = temp[BANK_OFFSET(c)] + add - e3;
    temp[BANK_OFFSET(d)] = temp[BANK_OFFSET(d)] + add - e4;
    if (thid == TOTAL_THREADS - 1) *sum = temp[BANK_OFFSET(d)] + e4;
    __syncthreads();
}

__device__ __inline__ void prefix_sum_warp_2elements(int thid, int* temp, int N)
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

static inline int nextPow2_2elements(int n)
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
__device__ __inline__ void scan_2elements(int thid, int N, int* temp, int* device_blocksum_2elements)
{
    int t;

    int a1;
    int b1;
    if (thid < TOTAL_THREADS)  // 512
    {
        a1 = (2 * thid + 1) - 1;
        b1 = (2 * thid + 2) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
    }
    __syncthreads();
    if (thid<TOTAL_THREADS>> 1)  // 256
    {
        a1 = ((2 * thid + 1) << 1) - 1;
        b1 = ((2 * thid + 2) << 1) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
    }
    __syncthreads();
    if (thid<TOTAL_THREADS>> 2)  // 128
    {
        a1 = ((2 * thid + 1) << 2) - 1;
        b1 = ((2 * thid + 2) << 2) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
    }
    __syncthreads();
    if (thid<TOTAL_THREADS>> 3)  // 64
    {
        a1 = ((2 * thid + 1) << 3) - 1;
        b1 = ((2 * thid + 2) << 3) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
    }
    if (thid<TOTAL_THREADS>> 4)  // 32
    {
        a1 = ((2 * thid + 1) << 4) - 1;
        b1 = ((2 * thid + 2) << 4) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
    }
    if (thid<TOTAL_THREADS>> 5)  // 16
    {
        a1 = (((thid << 1) + 1) << 5) - 1;
        b1 = (((thid << 1) + 2) << 5) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
    }
    if (thid<TOTAL_THREADS>> 6)  // 8
    {
        a1 = (((thid << 1) + 1) << 6) - 1;
        b1 = (((thid << 1) + 2) << 6) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
    }
    if (thid<TOTAL_THREADS>> 7)  // 4
    {
        a1 = (((thid << 1) + 1) << 7) - 1;
        b1 = (((thid << 1) + 2) << 7) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
    }
    if (thid<TOTAL_THREADS>> 8)  // 2
    {
        a1 = (((thid << 1) + 1) << 8) - 1;
        b1 = (((thid << 1) + 2) << 8) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        temp[b1] += temp[a1];
    }
    if (thid == 0)
    {
        *device_blocksum_2elements = temp[BANK_OFFSET(N - 1)];
        temp[BANK_OFFSET(N - 1)] = 0;
    }

    if (thid<TOTAL_THREADS>> 8)  // 2
    {
        int a1 = (((thid << 1) + 1) << 8) - 1;
        int b1 = (((thid << 1) + 2) << 8) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
    }
    if (thid<TOTAL_THREADS>> 7)  // 4
    {
        int a1 = (((thid << 1) + 1) << 7) - 1;
        int b1 = (((thid << 1) + 2) << 7) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
    }
    if (thid<TOTAL_THREADS>> 6)  // 8
    {
        int a1 = (((thid << 1) + 1) << 6) - 1;
        int b1 = (((thid << 1) + 2) << 6) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
    }
    if (thid<TOTAL_THREADS>> 5)  // 16
    {
        int a1 = (((thid << 1) + 1) << 5) - 1;
        int b1 = (((thid << 1) + 2) << 5) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
    }
    if (thid<TOTAL_THREADS>> 4)  // 32
    {
        int a1 = (((thid << 1) + 1) << 4) - 1;
        int b1 = (((thid << 1) + 2) << 4) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
    }
    if (thid<TOTAL_THREADS>> 3)  // 64
    {
        int a1 = (((thid << 1) + 1) << 3) - 1;
        int b1 = (((thid << 1) + 2) << 3) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
    }
    __syncthreads();
    if (thid<TOTAL_THREADS>> 2)  // 128
    {
        int a1 = (((thid << 1) + 1) << 2) - 1;
        int b1 = (((thid << 1) + 2) << 2) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
    }
    __syncthreads();
    if (thid<TOTAL_THREADS>> 1)  // 256
    {
        int a1 = (((thid << 1) + 1) << 1) - 1;
        int b1 = (((thid << 1) + 2) << 1) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
    }
    __syncthreads();
    if (thid < TOTAL_THREADS)  // 512
    {
        int a1 = (((thid << 1) + 1)) - 1;
        int b1 = (((thid << 1) + 2)) - 1;
        a1 = BANK_OFFSET(a1);
        b1 = BANK_OFFSET(b1);
        t = temp[a1];
        temp[a1] = temp[b1];
        temp[b1] += t;
    }
    __syncthreads();
}

__global__ void add_kernel_2elements(int* device_result, int* device_blocksum_2elements)
{
    __shared__ int temp1;
    int thid = threadIdx.x;
    int N = blockDim.x;
    if (thid == 0) temp1 = device_blocksum_2elements[blockIdx.x];
    __syncthreads();
    device_result[blockIdx.x * 4 * blockDim.x + thid] = device_result[blockIdx.x * 4 * blockDim.x + thid] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + N] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] + temp1;
    device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] =
        device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] + temp1;
}

__global__ void sweep_kernel_2elements(int N, int* device_x, int* device_result, int* device_blocksum_2elements)
{
    extern __shared__ int temp[];
    int thid = threadIdx.x;

    int a = thid;
    int b = thid + N / 4;
    int c = thid + 2 * N / 4;
    int d = thid + 3 * N / 4;
    a = BANK_OFFSET(a);
    b = BANK_OFFSET(b);
    c = BANK_OFFSET(c);
    d = BANK_OFFSET(d);
    temp[a] = device_result[blockIdx.x * 4 * blockDim.x + thid];
    temp[b] = device_result[blockIdx.x * 4 * blockDim.x + thid + N / 4];
    temp[c] = device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N / 4];
    temp[d] = device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N / 4];
    __syncthreads();

    scan_4bit(thid, temp, &device_blocksum_2elements[blockIdx.x]);

    device_result[blockIdx.x * 4 * blockDim.x + thid] = temp[a];
    device_result[blockIdx.x * 4 * blockDim.x + thid + N / 4] = temp[b];
    device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N / 4] = temp[c];
    device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N / 4] = temp[d];
}
__device__ __inline__ void hist_calc_2elements(int thid, int* sm, int nibble, int a, int b, int* hist, int sum)
{
    __shared__ int sum1;
    __shared__ int bit_vect[BANK_OFFSET(2 * TOTAL_THREADS)];
    __shared__ int bit_vect_p[BANK_OFFSET(2 * TOTAL_THREADS)];
    __shared__ int temp_hist[TWO_PWR(ITER)];

    bit_vect[BANK_OFFSET(a)] = !(((sm[BANK_OFFSET1(a)] >> (nibble * 4)) & (TWO_PWR(ITER) - 1)) ==
                                 ((sm[BANK_OFFSET1(b)] >> (nibble * 4)) & (TWO_PWR(ITER) - 1)));
    bit_vect[BANK_OFFSET(b)] = ((b + 1) < (2 * TOTAL_THREADS))
                                   ? (!(((sm[BANK_OFFSET1(b)] >> (nibble * 4)) & (TWO_PWR(ITER) - 1)) ==
                                        ((sm[BANK_OFFSET1(b + 1)] >> (nibble * 4)) & (TWO_PWR(ITER) - 1))))
                                   : 1;

    bit_vect_p[BANK_OFFSET(a)] = bit_vect[BANK_OFFSET(a)];
    bit_vect_p[BANK_OFFSET(b)] = bit_vect[BANK_OFFSET(b)];

    __syncthreads();

    scan_2elements(thid, 2 * TOTAL_THREADS, bit_vect_p, &sum1);

    if (bit_vect[BANK_OFFSET(a)] == 1) temp_hist[bit_vect_p[BANK_OFFSET(a)]] = a + 1;
    if (bit_vect[BANK_OFFSET(b)] == 1) temp_hist[bit_vect_p[BANK_OFFSET(b)]] = b + 1;
    __syncthreads();

    if (thid < 32)
    {
        if (thid > 0 && thid <= sum1)
        {
            temp_hist[thid] -= temp_hist[thid - 1];
        }
    }
    __syncthreads();

    if (bit_vect[BANK_OFFSET(a)] == 1)
        hist[(sm[BANK_OFFSET1(a)] >> (nibble * 4)) & (TWO_PWR(ITER) - 1)] = temp_hist[bit_vect_p[BANK_OFFSET(a)]];

    if (bit_vect[BANK_OFFSET(b)] == 1)
        hist[(sm[BANK_OFFSET1(b)] >> (nibble * 4)) & (TWO_PWR(ITER) - 1)] = temp_hist[bit_vect_p[BANK_OFFSET(b)]];
    __syncthreads();
}

__global__ void tile_sort_2elements(int nibble, int* device_input, int length, int num_blocks, int* device_hist,
                                    int* pdevice_hist)
{
    __shared__ int sm[BANK_OFFSET(2 * TOTAL_THREADS)];
    __shared__ int temp[BANK_OFFSET(2 * TOTAL_THREADS)];
    __shared__ int hist[TWO_PWR(ITER)];
    __shared__ int sum;
    int t = 2 * blockIdx.x * blockDim.x + 2 * threadIdx.x;
    int thid = threadIdx.x;
    if ((2 * thid + 1) < TWO_PWR(ITER))
    {
        hist[2 * thid] = 0;
        hist[2 * thid + 1] = 0;
    }
    __syncthreads();

    int a = 2 * thid;
    int b = 2 * thid + 1;
    if ((t + 1) < length)
    {
        sm[BANK_OFFSET1(a)] = device_input[t];
        sm[BANK_OFFSET1(b)] = device_input[t + 1];
    }
    else if (t < length)
    {
        sm[BANK_OFFSET(a)] = device_input[t];
        sm[BANK_OFFSET(b)] = INT_MAX;
    }
    else
    {
        sm[BANK_OFFSET(a)] = INT_MAX;
        sm[BANK_OFFSET(b)] = INT_MAX;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ITER; i++)
    {
        int cur_1 = sm[BANK_OFFSET1(a)];
        int cur_2 = sm[BANK_OFFSET1(b)];
        int bit1;
        int bit2;
        bit1 = temp[BANK_OFFSET(a)] = (sm[BANK_OFFSET1(a)] >> (i + nibble * 4)) & 0x1;
        bit2 = temp[BANK_OFFSET(b)] = (sm[BANK_OFFSET1(b)] >> (i + nibble * 4)) & 0x1;
        __syncthreads();
        scan_2elements(thid, 2 * TOTAL_THREADS, temp, &sum);
        int num_0 = 2 * TOTAL_THREADS - sum;
        if (bit1)
            sm[BANK_OFFSET1(temp[BANK_OFFSET(a)] + num_0)] = cur_1;
        else
            sm[BANK_OFFSET1(a - temp[BANK_OFFSET(a)])] = cur_1;
        if (bit2)
            sm[BANK_OFFSET1(temp[BANK_OFFSET(b)] + num_0)] = cur_2;
        else
            sm[BANK_OFFSET1(b - temp[BANK_OFFSET(b)])] = cur_2;
        __syncthreads();
    }

    hist_calc_2elements(thid, sm, nibble, a, b, hist, sum);
    device_input[t] = sm[BANK_OFFSET1(a)];
    device_input[t + 1] = sm[BANK_OFFSET1(b)];
    if ((2 * thid + 1) < TWO_PWR(ITER))
    {
        device_hist[2 * thid * num_blocks + blockIdx.x] = hist[2 * thid];
        device_hist[(2 * thid + 1) * num_blocks + blockIdx.x] = hist[2 * thid + 1];
        pdevice_hist[2 * thid * num_blocks + blockIdx.x] = hist[2 * thid];
        pdevice_hist[(2 * thid + 1) * num_blocks + blockIdx.x] = hist[2 * thid + 1];
    }
}

__global__ void output_index_2elements(int* device_hist, int* pdevice_hist, int* device_input, int* device_out,
                                       int length, int num_blocks, int nibble)
{
    __shared__ int temp[TWO_PWR(ITER)];
    int t = 2 * blockIdx.x * blockDim.x + 2 * threadIdx.x;
    int thid = threadIdx.x;
    int val1;
    int val2;
    int lindex1;
    int lindex2;
    int gindex1;
    int gindex2;
    int a = t;
    int b = t + 1;
    val1 = INT_MAX;
    val2 = INT_MAX;
    if (a < length) val1 = device_input[a];
    if (b < length) val2 = device_input[b];

    if (thid < 32)
    {
        if ((2 * thid + 1) < TWO_PWR(ITER))
        {
            temp[2 * thid] = device_hist[2 * thid * num_blocks + blockIdx.x];
            temp[2 * thid + 1] = device_hist[(2 * thid + 1) * num_blocks + blockIdx.x];
        }
        prefix_sum_warp_2elements(thid, temp, TWO_PWR(ITER));
    }
    __syncthreads();
    lindex1 = temp[((val1 >> (nibble * 4)) % (1 << ITER))];
    lindex2 = temp[((val2 >> (nibble * 4)) % (1 << ITER))];
    gindex1 = pdevice_hist[((val1 >> (nibble * 4)) % (1 << ITER)) * num_blocks + blockIdx.x];
    gindex2 = pdevice_hist[((val2 >> (nibble * 4)) % (1 << ITER)) * num_blocks + blockIdx.x];
    int idx1 = 2 * thid + gindex1 - lindex1;
    int idx2 = 2 * thid + 1 + gindex2 - lindex2;
    if (idx1 < length) device_out[idx1] = val1;
    if (idx2 < length) device_out[idx2] = val2;
}

int** device_blocksum_2elements;
int total_levels_2elements;

void alloc_blocksum_2elements(int N, int tpb)
{
    int level = 0;
    int m = N;
    do
    {
        m = (m + 4 * tpb - 1) / (4 * tpb);
        level++;
    } while (m > 1);
    device_blocksum_2elements = (int**)malloc(level * sizeof(int*));
    total_levels_2elements = level;
    level = 0;
    m = N;
    do
    {
        m = (m + 4 * tpb - 1) / (4 * tpb);
        cudaMalloc((void**)&device_blocksum_2elements[level], sizeof(int) * m);
        level++;
    } while (level < total_levels_2elements);
}
void dealloc_blocksum_2elements()
{
    int level = 0;
    for (level = 0; level < total_levels_2elements; level++)
    {
        cudaFree(device_blocksum_2elements[level]);
    }
    free(device_blocksum_2elements);
}

void exclusive_scan_2elements(int* device_start, int length, int* device_result, int level)
{
    int N = nextPow2_2elements(length);
    const int threadsPerBlock = TOTAL_THREADS;
    const int blocks = (N + 4 * threadsPerBlock - 1) / (4 * threadsPerBlock);

    if (level == 0)
    {
        sweep_kernel_2elements<<<blocks, threadsPerBlock, (BANK_OFFSET(4 * threadsPerBlock)) * sizeof(int)>>>(
            4 * threadsPerBlock, device_start, device_result, device_blocksum_2elements[level]);
    }
    else
    {
        sweep_kernel_2elements<<<blocks, threadsPerBlock, (BANK_OFFSET(4 * threadsPerBlock)) * sizeof(int)>>>(
            4 * threadsPerBlock, device_start, device_blocksum_2elements[level - 1], device_blocksum_2elements[level]);
    }
    if (blocks > 1)
    {
        exclusive_scan_2elements(device_start, blocks, device_blocksum_2elements[level], level + 1);
        if (level > 0)
        {
            add_kernel_2elements<<<blocks, threadsPerBlock, sizeof(int)>>>(device_blocksum_2elements[level - 1],
                                                                           device_blocksum_2elements[level]);
        }
        else
        {
            add_kernel_2elements<<<blocks, threadsPerBlock, sizeof(int)>>>(device_result,
                                                                           device_blocksum_2elements[level]);
        }
    }
}

void radix_sort_2elements(int* device_start, int length, int* device_result, int num_blocks, int* device_hist,
                          int* pdevice_hist)
{
    for (int nibble = 0; nibble < 32 / ITER; nibble++)
    {
        if (nibble % 2 == 0)
        {
            tile_sort_2elements<<<num_blocks, TOTAL_THREADS>>>(nibble, device_start, length, num_blocks, device_hist,
                                                               pdevice_hist);
            exclusive_scan_2elements(pdevice_hist, num_blocks * TWO_PWR(ITER), pdevice_hist, 0);
            output_index_2elements<<<num_blocks, TOTAL_THREADS>>>(device_hist, pdevice_hist, device_start,
                                                                  device_result, length, num_blocks, nibble);
        }
        else

        {
            tile_sort_2elements<<<num_blocks, TOTAL_THREADS>>>(nibble, device_result, length, num_blocks, device_hist,
                                                               pdevice_hist);
            exclusive_scan_2elements(pdevice_hist, num_blocks * TWO_PWR(ITER), pdevice_hist, 0);
            output_index_2elements<<<num_blocks, TOTAL_THREADS>>>(device_hist, pdevice_hist, device_result,
                                                                  device_start, length, num_blocks, nibble);
        }
    }
}

double cudaSort_2elements(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int rounded_length = nextPow2_2elements(end - inarray);
    int *device_hist, *pdevice_hist;
    int num_blocks = (rounded_length + 2 * TOTAL_THREADS - 1) / (2 * TOTAL_THREADS);
    int len = num_blocks * TWO_PWR(ITER);
    cudaMalloc((void**)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_input, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_hist, sizeof(int) * num_blocks * TWO_PWR(ITER));
    cudaMalloc((void**)&pdevice_hist, sizeof(int) * num_blocks * TWO_PWR(ITER));
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    alloc_blocksum_2elements(nextPow2_2elements(len), TOTAL_THREADS);
    double startTime = CycleTimer::currentSeconds();
    radix_sort_2elements(device_input, end - inarray, device_result, num_blocks, device_hist, pdevice_hist);
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);
    dealloc_blocksum_2elements();
    cudaFree(device_hist);
    cudaFree(pdevice_hist);
    return overallDuration;
}
