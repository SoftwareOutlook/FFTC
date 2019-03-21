CXX=g++
CC=gcc
FLAGS=  -L${OPT}/lib -I${OPT}/include -L/work/c01/c01/gambron/miniconda3/lib  -I${CASA}/include -std=c++1z -g2  -fopenmp   -lm -lfftw3 -lfftw3_threads -lgslcblas -lgsl -lboost_system -lboost_chrono -liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64  -lcasa_scimath  -llapack 


all: test

test: main.o fftpack.o
	${CXX} fftpack.o main.cpp -o test ${FLAGS}  

main.o: main.cpp complex.hpp multiarray.hpp signal.hpp stopwatch.hpp fft.hpp
	${CXX} -c main.cpp ${FLAGS} 

fftpack.o: fftpack.h fftpack.c
	${CC} -c fftpack.c -I${CASA}/include
clean:
	rm *.o test
