CXX=mpic++ #g++
CC=gcc
#FLAGS= -L/opt/cray/fftw/3.3.4.11/ivybridge/lib -L${OPT}/lib -I${OPT}/include -L/work/c01/c01/gambron/miniconda3/lib  -I${CASA}/include -std=c++1z -g2  -fopenmp   -lm -lfftw3 -lfftw3_threads -lfftw3_mpi -lgslcblas -lgsl -lboost_system -lboost_chrono -liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64  -lcasa_scimath  -llapack -pthread -lmpi

FLAGS= -L/opt/cray/fftw/3.3.4.11/ivybridge/lib -L$DATA/fftw/include -I$DATA/fftw/include  -std=c++1z -g2  -fopenmp   -lm -lfftw3 -lfftw3_threads -lfftw3_mpi -lgslcblas -lgsl -lboost_system -lboost_chrono -liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64  -llapack -pthread -lmpi




all: testmpi

#test: main.o fftpack.o
#	${CXX} fftpack.o main.cpp -o test ${FLAGS}  

testmpi: mainmpi.o 
	${CXX} mainmpi.cpp -o testmpi ${FLAGS}  

mainmpi.o: mainmpi.cpp complex.hpp multiarray.hpp signal.hpp stopwatch.hpp fft.hpp fftmpi.hpp
	${CXX} -c mainmpi.cpp ${FLAGS}

#main.o: main.cpp complex.hpp multiarray.hpp signal.hpp stopwatch.hpp fft.hpp fftmpi.hpp
#	${CXX} -c main.cpp ${FLAGS} 

clean:
	rm *.o test testmpi
