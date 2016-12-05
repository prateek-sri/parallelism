#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <iostream>
#include <omp.h>
#include "CycleTimer.h"

#define SIZE 16777216
static int scan (int * shared, int *arr, int size)
{
    int num_thd, tid;
    int64_t slice_len,slice_begin, slice_end, t1, t2, k;
    num_thd = omp_get_num_threads ();
    tid = omp_get_thread_num ();
    t1 = size / (num_thd);
    t2 = size % (num_thd);
    slice_begin = t1 * tid + (tid < t2? tid : t2);
    slice_end = t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);
    shared[tid] = 0;
    for (k = slice_begin; k < slice_end; ++k)
       shared[tid] += arr[k];
    int64_t ret;
#pragma omp barrier
#pragma omp single
    {
        int64_t tmp = shared[0];
        int64_t cur;
        shared[0] =0;
        for (k=1; k<=num_thd; k++)
        {
            cur = shared[k];
            shared[k]=tmp+shared[k-1];
            tmp = cur;
        }
    }
#pragma omp barrier
    t1 = shared[tid];
    for (k = slice_begin ; k < slice_end; ++k) {
        int64_t tmp = arr[k];
        arr[k] = t1;
        t1 += tmp;
    }
    return shared[num_thd];
}

void printarray(int arr[], int size)
{
    for(int i=0;i<size;i++)
        std::cout<<arr[i]<<" ";
    std::cout<<std::endl;
}

int prefix_sum(int* arr, int size)
{
    for(int i=1;i<size;i++)
    {
        arr[i]+=arr[i-1];
    }

    int val=arr[size-1];
    for(int i=size-1;i>0;i--)
    {
        arr[i]=arr[i-1];
    }
    arr[0]=0;
    return val;
}

void  radixsortkernel(int *arr, int *temp, int size, int bit_cur, int* bins)
{
    for (int bit = bit_cur; bit < bit_cur + 4; bit++) 
    {
	int bin[2];
        if(bit % 2 == 0)
        {
            bin[0] = 0;
            for (int i = 0; i < size; i++)
            {
                if (((arr[i] >> bit ) & 1) == 0)
                    bin[0]++;
            }

            bin[1] = size;

            for (int i = (size - 1); i >= 0; i--) 
                temp[--bin[((arr[i] >> bit) & 1)]] = arr[i];
        }
        else
        {
            bin[0] = 0;
            for (int i = 0; i < size; i++)
            {
                if (((temp[i] >> bit ) & 1) == 0)
                    bin[0]++;
            }

            bin[1] = size;

            for (int i = (size - 1); i >= 0; i--) 
                arr[--bin[(temp[i] >> bit) & 1]] = temp[i];
        }
    }
    int last = 0;
    for (int i =0; i < size; i++)
        temp[i] = (arr[i] >> bit_cur) & 0xf;

    for(int i = 0; i < 16; i++)
        bins[i] =0;

    for(int i = 0; i < (size - 1); i++)
    {
        if (temp[i] != temp[i+1])
        {
            bins[temp[i]] = i + 1 - last;
            last += bins[temp[i]];
        }
    }

    bins[temp[size - 1]] = (temp[size - 2] == temp[size - 1]) ? size - last : 1;
}

void radixsort(int* arr, int* arr_odd, int l)
{
    int i;
    int *bins, *buffer, *temp_bins;
    int bit = 0;
    int temp[l]; 
    int max_threads=omp_get_max_threads();
    bins=(int*)malloc(16*max_threads*sizeof(int));
    temp_bins=(int*)malloc(16*max_threads*sizeof(int));
    buffer=(int*)malloc(max_threads*sizeof(int));

    int total;

    for (bit =0; bit <32; bit+=4)
    {
#pragma omp parallel
        {
            int num_thd=omp_get_num_threads();
            int id=omp_get_thread_num();
            int t1 = l / (num_thd);
            int t2 = l % (num_thd);
            int slice_begin = t1 * id + (id < t2? id : t2);
            int slice_end = t1 * (id+1) + ((id+1) < t2? (id+1) : t2);
            //    std::cout<<slice_begin<<" - "<<slice_end<<std::endl;
            if (bit % 8 == 0)
		radixsortkernel(&arr[slice_begin], &temp[slice_begin], slice_end - slice_begin, bit, &bins[id*16]);
            else
		radixsortkernel(&arr_odd[slice_begin], &temp[slice_begin], slice_end - slice_begin, bit, &bins[id*16]);
#pragma omp barrier
#pragma omp single
            {
                //printarray(bins,16);
                for(int j = 0; j < num_thd*16; j++)
                {
                    //    printf("%d,", (j/256 + (j%256) * 16));
                    temp_bins[j] = bins[j/256 + (j%256) * 16];
                }
                total = prefix_sum(temp_bins,16*num_thd);
            }
#pragma omp barrier
            //start_1 = total + (slice_begin - bins[id]) - val;
            //for(int j=0;j<val;j++)
            //{
            //    arr[j+bins[id]]=temp[slice_begin+j];
            //}
            prefix_sum(&bins[id*16], 16);

            for(int j = 0;j < (slice_end - slice_begin); j++)
            {
                int gindex = temp_bins[id + 256 * temp[slice_begin + j]];
                int lindex = bins[id * 16 + temp[slice_begin + j]];
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
