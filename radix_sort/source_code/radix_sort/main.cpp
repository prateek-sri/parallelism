#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>

#include "CycleTimer.h"

double cudaSort(int* start, int* end, int* resultarray);
double cudaSort_2elements(int* start, int* end, int* resultarray);
void printCudaInfo();

void usage(const char* progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -t  --thrust           Use Thrust library implementation\n");
    printf("  -?  --help             This message\n");
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
        bit *= 10;
    }
}

int main(int argc, char** argv)
{
    int N = 64;
    bool useThrust = false;
    std::string test;
    std::string input;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"arraysize", 1, 0, 'n'}, {"help", 0, 0, '?'}, {"thrust", 0, 0, 't'}, {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "m:n:i:?t", long_options, NULL)) != EOF)
    {
        switch (opt)
        {
            case 'n':
                N = atoi(optarg);
                break;
            case 't':
                useThrust = true;
                break;
            case '?':
            default:
                usage(argv[0]);
                return 1;
        }
        test = "sort";
        input = "random";
    }
    // end parsing of commandline options //////////////////////////////////////

    int* inarray = new int[N];
    int* resultarray = new int[N];
    int* checkarray = new int[N];

    srand(time(NULL));
    // generate random array
    for (int i = 0; i < N; i++)
    {
        int val = rand() % 200;
        inarray[i] = val;
        checkarray[i] = val;
    }

    printCudaInfo();

    double cudaTime = 50000.;
    double cudaTime_2element = 50000.;
    double serialTime = 50000.;

    if (test.compare("sort") == 0)
    {
        for (int i = 0; i < 1; i++)
        {
            cudaTime_2element = std::min(cudaTime_2element, cudaSort_2elements(inarray, inarray + N, resultarray));
            cudaTime = std::min(cudaTime, cudaSort(inarray, inarray + N, resultarray));
            cudaTime_2element = 1000.f * cudaTime_2element;
            cudaTime = 1000.f * cudaTime;
        }

        for (int i = 0; i < 1; i++)
        {
            double startTime = CycleTimer::currentSeconds();
            radixsort_CPU(checkarray, N);
            double endTime = CycleTimer::currentSeconds();
            serialTime = std::min(serialTime, endTime - startTime);
        }

        if (useThrust)
        {
            printf("Thrust GPU time: %.3f ms\n", 1000.f * cudaTime);
        }
        else
        {
            printf("\nCPU_time: %.3f ms\n", 1000.f * serialTime);
            printf("\n---------------------------------------------------------\n");
            printf("\nParallel Radix sort using 4 elements per thread\n");
            printf("GPU_time: %.3f ms\n", cudaTime);
            printf("GPU_speedup: %.3fx times serial implementation\n", 1000.f * serialTime / cudaTime);

            printf("\n---------------------------------------------------------\n");
            printf("\nParallel Radix sort using 2 elements per thread\n");
            printf("GPU_time: %.3f ms\n", cudaTime_2element);
            printf("GPU_speedup: %.3fx times serial implementation\n", 1000.f * serialTime / cudaTime_2element);
            printf("\n---------------------------------------------------------\n\n");
        }

        // validate results
        for (int i = 0; i < N; i++)
        {
            if (checkarray[i] != resultarray[i])
            {
                fprintf(stderr,
                        "Error: Device Radix sort outputs incorrect result."
                        " A[%d] = %d, expecting %d.\n",
                        i, resultarray[i], checkarray[i]);
                exit(1);
            }
        }
        printf("Radix Sort outputs are correct!\n");
    }

    delete[] inarray;
    delete[] resultarray;
    delete[] checkarray;
    return 0;
}
