#!/bin/bash
#SBATCH -J sort           # job name
#SBATCH -o sort.o%j       # output and error file name (%j expands to jobID)
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p gpudev     # queue (partition) -- normal, development, etc.
#SBATCH -t 01:30:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH -A Fa16EE382N-20
./cudaSort -n 16777216
