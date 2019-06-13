CXX=CC

MKL_FLAGS=-O3 -DMKL_ILP64 -openmp -I${MKLROOT}/include -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_cdft_core.a ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

FLAGS=-lboost_system -lboost_timer -lboost_chrono -lrt -lfftw3
all: testfftw testmkl

testmkl: mainmkl.o 
	${CXX} mainmkl.cpp -o testmkl ${FLAGS} ${MKL_FLAGS}

mainmkl.o: mainmkl.cpp complex.hpp stopwatch.hpp 
	${CXX} -c mainmkl.cpp ${FLAGS} ${MKL_FLAGS}

testfftw: mainfftw.o 
	${CXX} mainfftw.cpp -o testfftw ${FLAGS} ${MKL_FLAGS} 

mainfftw.o: mainfftw.cpp complex.hpp stopwatch.hpp 
	${CXX} -c mainfftw.cpp ${FLAGS} ${MKL_FLAGS}

clean:
	rm *.o testfftw
