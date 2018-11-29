CXX=g++
CC=gcc
FLAGS=-std=c++11 -g2 -fopenmp -lm -lfftw3 -lfftw3_threads -lgslcblas -lgsl -lboost_system -lboost_chrono -liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64  -lcasa_scimath -llapack

all: test

test: main.cpp complex.hpp cube.hpp signal.hpp stopwatch.hpp fft.hpp fftpack.o
	${CXX} fftpack.o main.cpp -o test ${FLAGS}  

fftpack.o: fftpack.h fftpack.c
	${CC} -c fftpack.c
clean:
	rm *.o test
