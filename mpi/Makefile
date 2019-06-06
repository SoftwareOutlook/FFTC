CXX=mpic++
FLAGS=-std=c++1z -O3 -fopenmp -lm -lfftw3 -lfftw3_threads -lfftw3_mpi -lboost_system -lboost_chrono 


all: testfftw 

testfftw: mainfftw.o 
	${CXX} mainfftw.cpp -o testfftw ${FLAGS}  

mainfftw.o: mainfftw.cpp complex.hpp stopwatch.hpp 
	${CXX} -c mainfftw.cpp ${FLAGS} 

clean:
	rm *.o testfftw
