/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"


__global__ void block2D_reg_tiling(float c0,float c1,float *current,float *next, int nx, int ny, int nz)
{
    extern __shared__ float shared[];
    int size=blockDim.x*blockDim.y;
    int tx=blockDim.x;
    float * A1=&shared[0];
    float * A2=&A1[size];
    float * A3=&A1[2*size];
    float * Anew=&A1[3*size];
    float * temp;

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    if( i>0 && j>0 &&(i<nx-1) &&(j<ny-1) )
    {
        int id=i+tx*j;
        A1[id]=current[Index3D(nx, ny, i, j, 0)] ;
        A2[id]=current[Index3D (nx, ny, i, j, 1)] ;
        for(int k=1;k<nz-1;k++)
        {
            A3[id]=current[Index3D (nx, ny, i, j, k+1)] ;
            __syncthreads();
            Anew[id]=0;
            if(threadIdx.x==0)
                Anew[id]+=current[Index3D (nx, ny, i - 1, j, k)];
            else
                Anew[id]+=A2[i-1+tx*j];

            if(threadIdx.x==blockDim.x-1)
                Anew[id]+=current[Index3D (nx, ny, i + 1, j, k)];
            else
                Anew[id]+=A2[i+1+tx*j];

            if(threadIdx.y==0)
                Anew[id]+=current[Index3D (nx, ny, i , j - 1, k)];
            else
                Anew[id]+=A2[i+tx*(j-1)];
            
            if(threadIdx.y==blockDim.y-1)
                Anew[id]+=current[Index3D (nx, ny, i , j + 1, k)];
            else
                Anew[id]+=A2[i+tx*(j+1)];
            
            Anew[id] = c1*(A1[id] + A3[id] + Anew[id])- A2[id]*c0;
            next[Index3D (nx, ny, i, j, k)]=Anew[id]; 
            temp=A1;
            A1=A2;
            A2=A3;
            A3=temp;
            __syncthreads();
        }
    }

}


