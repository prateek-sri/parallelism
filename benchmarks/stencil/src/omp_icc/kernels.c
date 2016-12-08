/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"
#define LEN 4

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
#pragma omp parallel for collapse(3)
    for(int i = 1; i < nx - 1; i += LEN)
    {
        for(int j = 1; j < ny - 1; j += LEN)
        {
            for(int k = 1; k < nz - 1; k += LEN)
            {
                int i2_end = (i + LEN < nx - 1) ? i + LEN : nx - 1;
                int j2_end = (j + LEN < ny - 1) ? j + LEN : ny - 1;
                int k2_end = (k + LEN < nz - 1) ? k + LEN : nz - 1;
                for (int i2 = i; i2 < i2_end; i2++)
                    for (int j2 = j; j2 < j2_end; j2++)
                        for (int k2 = k; k2 < k2_end; k2++) {
                            Anext[Index3D (nx, ny, i2, j2, k2)] = 
                                (A0[Index3D (nx, ny, i2, j2, k2 + 1)] +
                                 A0[Index3D (nx, ny, i2, j2, k2 - 1)] +
                                 A0[Index3D (nx, ny, i2, j2 + 1, k2)] +
                                 A0[Index3D (nx, ny, i2, j2 - 1, k2)] +
                                 A0[Index3D (nx, ny, i2 + 1, j2, k2)] +
                                 A0[Index3D (nx, ny, i2 - 1, j2, k2)])*c1
                                - A0[Index3D (nx, ny, i2, j2, k2)]*c0;
                        }
            }
        }
    }
}
