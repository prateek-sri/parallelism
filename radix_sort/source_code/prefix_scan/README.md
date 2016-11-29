
/***************************************************/

PARALLEL GPU BASED SORTING ALGORTIHMS
IMPLEMENTATION OF PARALLEL PREFIX SCAN ON NVIDIA GPU

/***************************************************/

Instructions to compile and run Parallel Scan on Stampede

1. To build the code : Run the Makefile after navigating into the prefix_scan directory
$ make clean
$ make


2. To run the executable on Stampede using sbatch file :
$ sbatch scan.sbatch
NOTE : The size of input array can be varied by editing the sbatch file. 
The argument passed with option -n defines the size of array to be sorted.
The result will be in a scan.o<jobID> file.


3. To individually run the executable
$ ./cudaScan -n <size_of_input_array> -i random -m scan

/***************************************************/
