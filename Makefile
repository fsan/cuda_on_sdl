CXX=g++
NVCC=nvcc

CUDA_VERSION=10.0
INC_DIRS=/usr/local/cuda-${CUDA_VERSION}/include
INC=$(foreach d, $(INC_DIRS), -I$d)

all: main

main: main.o gpu.o
	${CXX} -o main main.o gpu.o -lSDL2 -lcudart

main.o: main.cpp
	${CXX} -c -o main.o main.cpp

gpu.o: gpu.cu gpu.h
	nvcc $(INC) -c -o gpu.o gpu.cu

clean: 
	rm -f *.o main
