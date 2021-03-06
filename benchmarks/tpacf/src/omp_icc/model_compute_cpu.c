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
#include <omp.h>

#include "model.h"

#define LEN 2048
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
#pragma omp parallel for 
    for (ii =0; ii < nbins+2; ii++)
        for (j = 0; j < 256; j++)
            data_bins1[j][ii] = 0;
    int i_end = doSelf ? n1-1 : n1;

    for (j = 0; j < n2; j+=LEN) 
    {
#pragma omp parallel for      
        for (i = 0; i < i_end; i++)
        {
            const register float xi = data1[i].x;
            const register float yi = data1[i].y;
            const register float zi = data1[i].z;
            int jj_start = (doSelf & (j < i + 1)) ? i + 1: j;
            int jj_end = (j + LEN < n2) ? j + LEN : n2;
            for (int jj = jj_start; jj < jj_end; jj++)
            {
                int tid = omp_get_thread_num ();
                register float dot = xi * data2[jj].x + yi * data2[jj].y + 
                    zi * data2[jj].z;

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
                }
                //#pragma omp atomic	  
                data_bins1[tid][max-1] ++; /*k = max;*/ 
            }
        }
    }
#pragma omp parallel for
    for (ii =0; ii < nbins+2; ii++)
        for (j = 0; j < 256; j++)
            data_bins[ii]+=data_bins1[j][ii];

    return 0;
}

