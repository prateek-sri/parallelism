# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# gcc (default)
CC = icc
PLATFORM_CFLAGS = -O3 -xMIC-AVX512 -DUSE_MMAP_LARGE -DUSE_MMAP_LARGE_EXT -qopenmp
  
CXX = icc
PLATFORM_CXXFLAGS = -O3 -xMIC-AVX512 -DUSE_MMAP_LARGE -DUSE_MMAP_LARGE_EXT -qopenmp

LINKER = icc
PLATFORM_LDFLAGS = -lm -lpthread -parallel -qopenmp

