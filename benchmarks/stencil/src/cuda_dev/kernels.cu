/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"

volatile __device__ int count = 0;
__device__ void global_barrier(int limit){
  __syncthreads();
  if(threadIdx.x == 0){
    atomicAdd((int*)&count, 1);
    while(count < limit){
      ;
    }
  }
  __syncthreads();

}

__global__ void block2D_reg_tiling(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz, int iteration)
{
    float *Acurr=A0;
    float *Anew=Anext;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    if( i>0 && j>0 &&(i<nx-1) &&(j<ny-1) )
    {
        for(int k=0;k<iteration;k++)
        {
            if(k%2)
            {
                Acurr=Anext;
                Anew=A0;
            }
            float bottom=Acurr[Index3D (nx, ny, i, j, 0)] ;
            float current=Acurr[Index3D (nx, ny, i, j, 1)] ;
            for(int k=1;k<nz-1;k++)
            {
                float top =Acurr[Index3D (nx, ny, i, j, k+1)] ;

                Anew[Index3D (nx, ny, i, j, k)] = 
                    (top +
                     bottom +
                     Acurr[Index3D (nx, ny, i, j + 1, k)] +
                     Acurr[Index3D (nx, ny, i, j - 1, k)] +
                     Acurr[Index3D (nx, ny, i + 1, j, k)] +
                     Acurr[Index3D (nx, ny, i - 1, j, k)])*c1
                    - current*c0;
                bottom=current;
                current=top;
            }
            global_barrier(gridDim.x*gridDim.y*gridDim.z);
        }
    }

}


