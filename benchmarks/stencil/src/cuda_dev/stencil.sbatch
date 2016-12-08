#!/bin/bash
#SBATCH -J saxpy           # job name
#SBATCH -o saxpy.o%j       # output and error file name (%j expands to jobID)
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p gpudev     # queue (partition) -- normal, development, etc.
#SBATCH -t 01:30:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=kshitiz@utexas.edu
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH -A Fa16EE382N-20
../../../../parboil run stencil cuda_fermi default
