CXX=CC

FLAGS=-DMKL_ILP64 -openmp -I${MKLROOT}/include -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a ${MKLROOT}/lib/intel64/libmkl_cdft_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl              -L/opt/intel/composer_xe_2015.2.164/compiler/lib/intel64 -lboost_system
all: testfftw testmkl

testmkl: mainmkl.o 
	${CXX} mainmkl.cpp -o testmkl ${FLAGS}  

mainmkl.o: mainmkl.cpp complex.hpp stopwatch.hpp 
	${CXX} -c mainmkl.cpp ${FLAGS} 

testfftw: mainfftw.o 
	${CXX} mainfftw.cpp -o testfftw ${FLAGS}  

mainfftw.o: mainfftw.cpp complex.hpp stopwatch.hpp 
	${CXX} -c mainfftw.cpp ${FLAGS} 

clean:
	rm *.o testfftw
