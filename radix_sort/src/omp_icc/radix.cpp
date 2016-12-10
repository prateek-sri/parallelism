#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include "CycleTimer.h"

#define SIZE  268435456 // limit to sum of values not exceeding INT_MAX in prefix sum
#define MAX_THREADS 256
#define NUM_BINS 16
#define MODULO(a, b) ((a) & ((b) - 1)) // only power of 2 will work
#define BLOCK_SIZE 16384
#define PREFIX_STEP 256
#define PREFIX_STEP_LOG 8

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
static void  prefix_sum(uint32_t* __restrict shared, uint32_t size)
{
    uint32_t tmp = shared[0];
    uint32_t cur;
    shared[0] =0;
    for (uint32_t k=1; k < size; k++)
    {
        cur = shared[k];
        shared[k]=tmp+shared[k-1];
        tmp = cur;
    }
}

static uint32_t  prefix_sum_r(uint32_t* __restrict shared, uint32_t size)
{
    uint32_t tmp = shared[0];
    uint32_t cur;
    shared[0] =0;
    for (uint32_t k=1; k < size; k++)
    {
        cur = shared[k];
        shared[k]=tmp+shared[k-1];
        tmp = cur;
    }
    return tmp + shared[size-1];
}

void printarray(uint32_t arr[], uint32_t size)
{
    for(uint32_t i=0;i<size;i++)
        std::cout<<arr[i]<<" ";
    std::cout<<std::endl;
}

void  radixsortkernel(uint32_t * __restrict arr, uint32_t * __restrict arr_4bit, uint32_t size, uint32_t bit_cur, uint32_t* __restrict  bins)
{
    uint32_t num_iter =  (size + BLOCK_SIZE - 1)/BLOCK_SIZE;
    
    uint32_t **bins_per_block;
    uint32_t *bins_sum =  (uint32_t*) malloc (sizeof(uint32_t) * num_iter * NUM_BINS);

    if (bins_sum == NULL)
    {
        printf("Out of memory");
        exit(1);
    }
    bins_per_block = (uint32_t**)malloc(sizeof(uint32_t*)*num_iter);

    if (bins_per_block == NULL)
    {
        printf("Out of memory");
        exit(1);
    }
    for (uint32_t iter = 0; iter < num_iter; iter++)
    {
        bins_per_block[iter] = (uint32_t *) malloc (sizeof (uint32_t) * NUM_BINS);
        if (bins_per_block[iter] == NULL)
        {
            printf("Out of memory");
            exit(1);
        }
        for (uint32_t bit = bit_cur; bit < bit_cur + 4; bit++) 
        {
            uint32_t bin[2];
            if((bit & 1) == 0)
            {
                bin[1] = 0;
                for (uint32_t i = iter * BLOCK_SIZE; i < ((( (iter+1) * BLOCK_SIZE) < size) ? (iter + 1) * BLOCK_SIZE : size); i++)
                { 
                    if (((arr[i] >> bit ) & 1) == 0)
                        bin[1]++;
                }

                bin[0] = 0;

                uint32_t k = iter * BLOCK_SIZE;
                uint32_t l = iter * BLOCK_SIZE;
                for (uint32_t i = iter * BLOCK_SIZE; i < ((( (iter+1) * BLOCK_SIZE) < size) ? (iter + 1) * BLOCK_SIZE : size); i++)
                {
                    uint32_t cur_bit = (arr[i] >> bit) & 1;
                    if (cur_bit == 0)
                        arr_4bit[k++] = arr[i];
                    else
                    {
                        arr_4bit[bin[1] +l] = arr[i];
                        l++;
                    }
                }
            }
            else
            {
                bin[1] = 0;
                for (uint32_t i = iter * BLOCK_SIZE; i < ((( (iter+1) * BLOCK_SIZE) < size) ? (iter + 1) * BLOCK_SIZE : size); i++)
                {
                    if (((arr_4bit[i] >> bit ) & 1) == 0)
                        bin[1]++;
                }

                bin[0] = 0;

                uint32_t k = iter * BLOCK_SIZE;
                uint32_t l = iter * BLOCK_SIZE;
                for (uint32_t i = iter * BLOCK_SIZE; i < ((( (iter+1) * BLOCK_SIZE) < size) ? (iter + 1) * BLOCK_SIZE : size); i++)
                {
                    uint32_t cur_bit = (arr_4bit[i] >> bit) & 1;
                    if (cur_bit == 0)
                        arr[k++] = arr_4bit[i];
                    else
                    {
                        arr[bin[1] +l] = arr_4bit[i];
                        l++;
                    }
                }
            }
        }

        for(uint32_t i = 0; i < NUM_BINS; i++)
            bins_per_block[iter][i] =0;

        for (uint32_t i = iter * BLOCK_SIZE; i < ((( (iter+1) * BLOCK_SIZE) < size) ? (iter + 1) * BLOCK_SIZE : size); i++)
        {
            (bins_per_block[iter][(arr[i] >> bit_cur) & 0xf])++;
        }
    }

    for (int i = 0; i < NUM_BINS; i++)
    {
        bins[i] = 0;
        for(uint32_t j = 0; j < num_iter; j++)
        {
            bins[i] += bins_per_block[j][i];
            bins_sum[i * num_iter + j] = bins_per_block[j][i];
        }
    }
    prefix_sum(bins_sum,NUM_BINS*num_iter);


    for (int i = 0; i < num_iter; i++)
    {
        prefix_sum(bins_per_block[i], NUM_BINS);
        for (uint32_t j = i * BLOCK_SIZE; j < ((( (i+1) * BLOCK_SIZE) < size) ? (i + 1) * BLOCK_SIZE : size); j++)
        {
            uint32_t gindex = bins_sum[i + num_iter * ((arr[j]>>bit_cur)&0xf)];
            uint32_t lindex = bins_per_block[i][((arr[j]>>bit_cur)&0xf)];
            arr_4bit[ gindex + j -i*BLOCK_SIZE - lindex] = arr[j];
        }
    }
}
unsigned int log2 (unsigned int v)
{
    static const unsigned int b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 
        0xFF00FF00, 0xFFFF0000};
    register unsigned int r = (v & b[0]) != 0;
    for (int i = 4; i > 0; i--) 
    {
        r |= ((v & b[i]) != 0) << i;
    }
    return r;
}
void radixsort(uint32_t* __restrict arr, uint32_t* __restrict arr_odd, uint32_t l)
{
    uint32_t * __restrict bins, * __restrict bins_sum;
    uint32_t bit = 0;
    uint32_t * temp = (uint32_t*)malloc(l*sizeof(uint32_t)); 
    uint32_t buffer[MAX_THREADS];
    bins=(uint32_t*)malloc(NUM_BINS*MAX_THREADS*sizeof(uint32_t));
    bins_sum=(uint32_t*)malloc(NUM_BINS*MAX_THREADS*sizeof(uint32_t));

    uint32_t size = (nextPow2_2elements(l) + MAX_THREADS-1)/MAX_THREADS;
    int shf = log2(size);
    for (bit =0; bit <32; bit+=4)
    {
        {
#pragma omp parallel for
            for (int i = 0; i < l; i+=size )
            {
                int k = i >> shf; 
                    radixsortkernel(&arr[i], &temp[i], (l-i)>size?size:l-i, bit, &bins[(k)*NUM_BINS]);
            }
#pragma omp parallel
            {
#pragma omp for
                for (int i = 0; i < NUM_BINS; i++)
                {
                    for(uint32_t j = i*MAX_THREADS; j < (i*MAX_THREADS + MAX_THREADS); j++)
                    {
                        bins_sum[j] = bins[j/MAX_THREADS + (MODULO(j, MAX_THREADS)) * NUM_BINS];
                    }
                    buffer[i] = prefix_sum_r(&bins_sum[i*MAX_THREADS], MAX_THREADS);
                }
#pragma omp barrier
#pragma omp single
                prefix_sum(buffer, NUM_BINS);
#pragma omp barrier
#pragma omp for
                for (int i = 0; i < MAX_THREADS*NUM_BINS; i++)
                {
                    bins_sum[i] += buffer[i>>PREFIX_STEP_LOG];
                }
            }

#pragma omp parallel for
            for (int i = 0; i < l; i+=size)
            {
                int k = i >> shf; 
                prefix_sum(&bins[(k)*NUM_BINS], NUM_BINS);
                for(uint32_t j = i; j < ((i+size)>l?l:i+size); j++)
                {
                    uint32_t gindex = bins_sum[(k) + MAX_THREADS * ((temp[j]>>bit)&0xf)];
                    uint32_t lindex = bins[(k) * NUM_BINS + ((temp[j]>>bit)&0xf)];
                    arr[ gindex + j - i - lindex] = temp[j];
                }
            }
        }
    }
}

void radixsort_CPU(uint32_t* arr, uint32_t l)
{
    uint32_t i, max = 0;
    uint32_t bit = 1;
    uint32_t bin[l], temp[l];

    for (i = 0; i < l; i++)
    {
        // Find the largest in the array
        if (arr[i] > max) max = arr[i];
    }

    while ((max / bit) > 0)
    {
        // initialize to 0
        for (i = 0; i < l; i++) bin[i] = 0;
        // Find the digit to be operated on starting from LSD
        for (i = 0; i < l; i++) bin[(arr[i] / bit) % 10]++;
        // Prefix sum on the digits
        for (i = 1; i < l; i++) bin[i] += bin[i - 1];
        // Sorted array in temp
        for (i = (l - 1); i >= 0; i--) temp[--bin[(arr[i] / bit) % 10]] = arr[i];
        // copy the sorted array in result
        for (i = 0; i < l; i++) arr[i] = temp[i];
        // digit scaling after every iteration
        bit *= 2;
    }
}

uint32_t main(uint32_t argc, char** argv)
{
    uint32_t array[SIZE];
    uint32_t array_odd[SIZE];
    uint32_t size=SIZE;
    srand(time(NULL));
    for(uint32_t i=0;i<size;i++)
    {
        array[i]=rand()%16;
        array_odd[i]=array[i];
    }
    printf("done input\n");
    printarray(array, 10);
    double serialTime = 0.0;
    double startTime = CycleTimer::currentSeconds();
    radixsort(array,array_odd, size);
    //radixsort_CPU(array, size);
    double endTime = CycleTimer::currentSeconds();
    serialTime = 1000.0 * (endTime - startTime);
    std::cout<<"Time : "<< serialTime <<" ms" << std::endl;
    uint32_t flag=0;
    for(uint32_t i=0;i<size-1;i++)
    {
        if(array[i]>array[i+1])
        {
            flag=1;
            printarray(array+i-1, 3);
            break;
        }
    }
    printarray(array, 10);
    if(flag==1)
    {
        std::cout<<"Fail:\n";
    }
    else
    {
        std::cout<<"Pass:\n";
    }
    return 0;
}
