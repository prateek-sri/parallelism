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
static void scan (uint32_t * __restrict shared, uint32_t * __restrict arr, uint32_t size)
{
    uint32_t  t1, k;
#pragma omp parallel for
    for (int i = 0; i < size; i+=16)
    {
        shared[i/16] = 0;
        for (k = i; k < i+16; ++k)
            shared[i/16] += arr[k];
    }
    uint32_t tmp = shared[0];
    uint32_t cur;
    shared[0] =0;
    for (k=1; k< MAX_THREADS; k++)
    {
        cur = shared[k];
        shared[k]=tmp+shared[k-1];
        tmp = cur;
    }
#pragma omp parallel for
    for (int i = 0; i < size; i+=16)
    {
        t1 = shared[i/16];
        for (k = i ; k <  i +16 ; ++k) {
            uint32_t tmp = arr[k];
            arr[k] = t1;
            t1 += tmp;
        }
    }
}

void printarray(uint32_t arr[], uint32_t size)
{
    for(uint32_t i=0;i<size;i++)
        std::cout<<arr[i]<<" ";
    std::cout<<std::endl;
}

void  prefix_sum(uint32_t* __restrict shared, uint32_t size)
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

void  radixsortkernel(uint32_t * __restrict arr, uint32_t * __restrict temp, uint32_t size, uint32_t bit_cur, uint32_t* __restrict  bins)
{
    for (uint32_t bit = bit_cur; bit < bit_cur + 4; bit++) 
    {
        uint32_t bin[2];
        if(bit % 2 == 0)
        {
            bin[1] = 0;
            for (uint32_t i = 0; i < size; i++)
            {
                if (((arr[i] >> bit ) & 1) == 0)
                    bin[1]++;
            }

            bin[0] = 0;

            uint32_t k = 0;
            uint32_t l = 0;
            for (uint32_t i = 0; i < size; i++)
            {
                uint32_t cur_bit = (arr[i] >> bit) & 1;
                if (cur_bit == 0)
                    temp[k++] = arr[i];
                else
                {
                    temp[bin[1] +l] = arr[i];
                    l++;
                }
            }
        }
        else
        {
            bin[1] = 0;
            for (uint32_t i = 0; i < size; i++)
            {
                if (((temp[i] >> bit ) & 1) == 0)
                    bin[1]++;
            }

            bin[0] = 0;

            uint32_t k = 0;
            uint32_t l = 0;
            for (uint32_t i = 0; i < size; i++)
            {
                uint32_t cur_bit = (temp[i] >> bit) & 1;
                if (cur_bit == 0)
                    arr[k++] = temp[i];
                else
                {
                    arr[bin[1] +l] = temp[i];
                    l++;
                }
            }
        }
    }
    uint32_t last = 0;
    for (uint32_t i =0; i < size; i++)
        temp[i] = (arr[i] >> bit_cur) & 0xf;

    for(uint32_t i = 0; i < NUM_BINS; i++)
        bins[i] =0;

    for(uint32_t i = 0; i < (size); i++)
    {
        bins[temp[i]]++;
        //if (temp[i] != temp[i+1])
        //{
        //    bins[temp[i]] = i + 1 - last;
        //    last += bins[temp[i]];
        //}
    }

    //bins[temp[size - 1]] = (temp[size - 2] == temp[size - 1]) ? size - last : 1;
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
    uint32_t * __restrict bins, * __restrict buffer, * __restrict temp_bins;
    uint32_t bit = 0;
    uint32_t temp[l]; 
    bins=(uint32_t*)malloc(NUM_BINS*MAX_THREADS*sizeof(uint32_t));
    temp_bins=(uint32_t*)malloc(NUM_BINS*MAX_THREADS*sizeof(uint32_t));
    buffer=(uint32_t*)malloc((MAX_THREADS)*sizeof(uint32_t));

            uint32_t size = nextPow2_2elements(l)/256;
            int shf = log2(size);
    for (bit =0; bit <32; bit+=4)
    {
        //#pragma omp parallel
        {
#pragma omp parallel for
            for (int i = 0; i < l; i+=size )
            {
               int k = i >> shf; 
                if (MODULO(bit, 8) == 0)
                    radixsortkernel(&arr[i], &temp[i], (l-i)>size?size:l-i, bit, &bins[(k)*NUM_BINS]);
                else
                    radixsortkernel(&arr_odd[i], &temp[i], (l-i)>size?size:l-i, bit, &bins[(k)*NUM_BINS]);

            }
            //#pragma omp barrier
#pragma omp parallel for
            for (int i = 0; i < NUM_BINS; i++)
            {
                for(uint32_t j = i*MAX_THREADS; j < (i*MAX_THREADS + MAX_THREADS); j++)
                {
                    temp_bins[j] = bins[j/MAX_THREADS + (MODULO(j, MAX_THREADS)) * NUM_BINS];
                }
            }
            //#pragma omp barrier
            scan(buffer,temp_bins,NUM_BINS*MAX_THREADS);
            //#pragma omp single
            //            {
            //                //printarray(bins,NUM_BINS);
            //                prefix_sum(temp_bins,NUM_BINS*MAX_THREADS);
            //            }
            //#pragma omp barrier

#pragma omp parallel for
            for (int i = 0; i < l; i+=size)
            {
                int k = i >> shf; 
                prefix_sum(&bins[(k)*NUM_BINS], NUM_BINS);
                for(uint32_t j = i; j < ((i+size)>l?l:i+size); j++)
                {
                    uint32_t gindex = temp_bins[(k) + MAX_THREADS * temp[j]];
                    uint32_t lindex = bins[(k) * NUM_BINS + temp[j]];
                    if (MODULO(bit, 8) == 0)
                        arr_odd[ gindex + j -i - lindex] = arr[j];
                    else
                        arr[ gindex + j - i - lindex] = arr_odd[j];
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
#pragma omp parallel for
    for(uint32_t i=0;i<size;i++)
    {
        array[i]=rand()%400000;
        array_odd[i]=array[i];
    }
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
        if(array_odd[i]>array_odd[i+1])
        {
            flag=1;
            printarray(array_odd+i-1, 3);
            break;
        }
    }
    printarray(array_odd, 10);
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
