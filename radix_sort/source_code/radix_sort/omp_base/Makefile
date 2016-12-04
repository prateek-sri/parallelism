CC = icc
CCFLAGS = -g -O3 -qopenmp -xMIC-AVX512 -qopenmp -DUSE_MMAP_LARGE -qopt-report=2 -DUSE_MMAP_LARGE_EXT -lrt 

all:
	@echo Running radix sort with $(OMP_NUM_THREADS) threads
	mkdir -p bin
	$(CC) $(CCFLAGS) radix.cpp -lm -o bin/radix
	./bin/radix
clean:
	rm -rf bin
