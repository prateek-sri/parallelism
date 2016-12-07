#!/bin/bash
#SBATCH -J sort           # job name
#SBATCH -o sort.o%j       # output and error file name (%j expands to jobID)
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p Flat-All2All     # queue (partition) -- normal, development, etc.
#SBATCH -t 00:05:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH -A Fa16EE382N-20
export OMP_NUM_THREADS=256
numactl --membind=1 ./bin/radix
