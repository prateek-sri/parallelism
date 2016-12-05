#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <iostream>
#include <omp.h>
#include "CycleTimer.h"

#define SIZE 33554432
#define MAX_THREADS 128
#define NUM_BINS 16
static void scan (int * __restrict shared, int * __restrict arr, int size)
{
    int num_thd, tid;
    int slice_len,slice_begin, slice_end, t1, t2, k;
    tid = omp_get_thread_num ();
    if (tid < MAX_THREADS/2)
    {
    slice_begin = 32*tid;
    slice_end = 32*tid +32;
    shared[tid] = 0;
    for (k = slice_begin; k < slice_end; ++k)
        shared[tid] += arr[k];
    }
#pragma omp barrier
#pragma omp single
    {
        int tmp = shared[0];
        int cur;
        shared[0] =0;
        for (k=1; k< MAX_THREADS/2; k++)
        {
            cur = shared[k];
            shared[k]=tmp+shared[k-1];
            tmp = cur;
        }
    }
#pragma omp barrier
    if (tid < MAX_THREADS/2)
    {
        t1 = shared[tid];
        for (k = slice_begin ; k < slice_end; ++k) {
            int tmp = arr[k];
            arr[k] = t1;
            t1 += tmp;
        }
    }
}

void printarray(int arr[], int size)
{
    for(int i=0;i<size;i++)
        std::cout<<arr[i]<<" ";
    std::cout<<std::endl;
}

void  prefix_sum(int* __restrict shared, int size)
{
    int tmp = shared[0];
    int cur;
    shared[0] =0;
    for (int k=1; k < size; k++)
    {
        cur = shared[k];
        shared[k]=tmp+shared[k-1];
        tmp = cur;
    }
}

void  radixsortkernel(int * __restrict arr, int * __restrict temp, int size, int bit_cur, int* __restrict  bins)
{
    for (int bit = bit_cur; bit < bit_cur + 4; bit++) 
    {
        int bin[2];
        if(bit % 2 == 0)
        {
            bin[1] = 0;
            for (int i = 0; i < size; i++)
            {
                if (((arr[i] >> bit ) & 1) == 0)
                    bin[1]++;
            }

            bin[0] = 0;

            int k = 0;
            int l = 0;
            for (int i = 0; i < size; i++)
            {
                int cur_bit = (arr[i] >> bit) & 1;
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
            for (int i = 0; i < size; i++)
            {
                if (((temp[i] >> bit ) & 1) == 0)
                    bin[1]++;
            }

            bin[0] = 0;

            int k = 0;
            int l = 0;
            for (int i = 0; i < size; i++)
            {
                int cur_bit = (temp[i] >> bit) & 1;
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
    int last = 0;
    for (int i =0; i < size; i++)
        temp[i] = (arr[i] >> bit_cur) & 0xf;

    for(int i = 0; i < NUM_BINS; i++)
        bins[i] =0;

    for(int i = 0; i < (size); i++)
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

void radixsort(int* __restrict arr, int* __restrict arr_odd, int l)
{
    int * __restrict bins, * __restrict buffer, * __restrict temp_bins;
    int bit = 0;
    int temp[l]; 
    bins=(int*)malloc(NUM_BINS*MAX_THREADS*sizeof(int));
    temp_bins=(int*)malloc(NUM_BINS*MAX_THREADS*sizeof(int));
    buffer=(int*)malloc((MAX_THREADS/2)*sizeof(int));

    for (bit =0; bit <32; bit+=4)
    {
#pragma omp parallel
        {
            int id=omp_get_thread_num();
            int t1 = l / (MAX_THREADS);
            int t2 = l % (MAX_THREADS);
            int slice_begin = t1 * id + (id < t2? id : t2);
            int slice_end = t1 * (id+1) + ((id+1) < t2? (id+1) : t2);
            //    std::cout<<slice_begin<<" - "<<slice_end<<std::endl;
            if (bit % 8 == 0)
                radixsortkernel(&arr[slice_begin], &temp[slice_begin], slice_end - slice_begin, bit, &bins[id*NUM_BINS]);
            else
                radixsortkernel(&arr_odd[slice_begin], &temp[slice_begin], slice_end - slice_begin, bit, &bins[id*NUM_BINS]);
#pragma omp barrier
            if(id <NUM_BINS)
            {
                for(int j = id*MAX_THREADS; j < (id*MAX_THREADS + MAX_THREADS); j++)
                {
                    temp_bins[j] = bins[j/MAX_THREADS + (j%MAX_THREADS) * NUM_BINS];
                }
            }
#pragma omp barrier
            scan(buffer,temp_bins,NUM_BINS*MAX_THREADS);
            //#pragma omp single
            //            {
            //                //printarray(bins,NUM_BINS);
            //                prefix_sum(temp_bins,NUM_BINS*MAX_THREADS);
            //            }
            //#pragma omp barrier
#pragma omp barrier
            //start_1 = total + (slice_begin - bins[id]) - val;
            //for(int j=0;j<val;j++)
            //{
            //    arr[j+bins[id]]=temp[slice_begin+j];
            //}
            prefix_sum(&bins[id*NUM_BINS], NUM_BINS);

            for(int j = 0;j < (slice_end - slice_begin); j++)
            {
                int gindex = temp_bins[id + MAX_THREADS * temp[slice_begin + j]];
                int lindex = bins[id * NUM_BINS + temp[slice_begin + j]];
                if (bit % 8 == 0)
                    arr_odd[ gindex + j - lindex] = arr[slice_begin + j];
                else
                    arr[ gindex + j - lindex] = arr_odd[slice_begin + j];
            }
        }
    }
}

void radixsort_CPU(int* arr, int l)
{
    int i, max = 0;
    int bit = 1;
    int bin[l], temp[l];

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

int main(int argc, char** argv)
{
    int array[SIZE];
    int array_odd[SIZE];
    int size=SIZE;
    for(int i=0;i<size;i++)
    {
        array[i]=(size-i-1);
        array_odd[i]=(size-i-1);
    }
    double serialTime = 0.0;
    double startTime = CycleTimer::currentSeconds();
    radixsort(array,array_odd, size);
    //radixsort_CPU(array, size);
    double endTime = CycleTimer::currentSeconds();
    serialTime = 1000.0 * (endTime - startTime);
    std::cout<<"Time : "<< serialTime <<" ms" << std::endl;
    int flag=0;
    for(int i=0;i<size;i++)
    {
        if(array_odd[i]!=i)
        {
            flag=1;
            break;
        }
    }
    //printarray(array, size);
    if(flag==1)
    {
        std::cout<<"Fail:\n";
    }
    else
    {
        std::cout<<"Pass:\n";
    }
}
