/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <stdio.h> 

#include "model.h"

int doCompute(struct cartesian *data1, int n1, struct cartesian *data2, 
        int n2, int doSelf, long long *data_bins, 
        int nbins, float *binb)
{
    int i, j, k, ii;
    long long data_bins1[256][nbins +2];
    if (doSelf)
    {
        n2 = n1;
        data2 = data1;
    }
    //  #pragma omp parallel for 
    for (ii =0; ii < nbins+2; ii++)
        data_bins[ii] = 0;
    for (ii =0; ii < nbins+2; ii++)
    {
        for (j = 0; j < 256; j++)
        {
            data_bins1[j][ii] = 0;
        }
    }
    for (i = 0; i < 1 /* ((doSelf) ? n1-1 : n1)*/; i++)
    {
        const register float xi = data1[i].x;
        const register float yi = data1[i].y;
        const register float zi = data1[i].z;

        int start  = ((doSelf) ? i+1 : 0); 
        int end = n2-1;
        int num_ele = end -start +1;
#pragma omp parallel
        {
            int num_thd = omp_get_num_threads ();
            printf("thid %d\n", num_thd);
            int tid = omp_get_thread_num ();
            int t1 = num_ele / (num_thd);
            int t2 = num_ele % (num_thd);
            int slice_begin = start + t1 * tid + (tid < t2? tid : t2);
            int slice_end = start + t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);
            printf("%d %d %d %d %d %d %d\n", start, end , slice_begin, slice_end, t1, t2, num_ele);
#pragma omp for      
            for (j = slice_begin; j < slice_end; j++)
            {
                register float dot = xi * data2[j].x + yi * data2[j].y + 
                    zi * data2[j].z;

                // run binary search
                register int min = 0;
                register int max = nbins + 2;
                register int k, indx;

                while (max > min + 1)
                {
                    k = (min + max) / 2;
                    if (dot >= binb[k]) 
                        max = k;
                    else 
                        min = k;
                };
                //#pragma omp atomic	  
                data_bins1[tid][max-1] += 1; /*k = max;*/ 

            }
        }
    }
            for (ii =0; ii < nbins+2; ii++)
            {
                for (j = 0; j < 256; j++)
                {
                    data_bins[ii]+=data_bins1[j][ii];
                }
            }

    return 0;
}

