CXX=g++
NVCC=nvcc

all: main

main: main.o
	${CXX} -o main main.o -lSDL2 && ./main

main.o: main.cpp
	${CXX} -c -o main.o main.cpp

clean: 
	rm -f *.o main
