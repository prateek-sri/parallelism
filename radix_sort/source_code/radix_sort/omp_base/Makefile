CC = icc
CCFLAGS = -g -O3 -qopenmp -qopt-report=2 -lrt 

all:
	$(CC) $(CCFLAGS) radix.cpp -lm -o radix
	#g++ radix.cpp -o radix
	./radix
clean:
	rm radix
