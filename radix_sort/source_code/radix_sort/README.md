
/***************************************************/

PARALLEL GPU BASED SORTING ALGORTIHMS
IMPLEMENTATION OF PARALLEL RADIX SORT ON NVIDIA GPU

/***************************************************/

Instructions to compile and run Radix Sort on Stampede

1. To build the code : Navigate into the code  directory and Run the Makefile 
$ make clean
$ make


2. To run the executable on Stampede using sbatch file :
$ sbatch sort.sbatch
NOTE : The size of input array can be varied by editing the sbatch file. 
The argument passed with option -n defines the size of array to be sorted.
The result will be in a sort.o<jobID> file.


3. To individually run the executable
$ ./cudaSort -n  <size_of_input_array>

/***************************************************/
