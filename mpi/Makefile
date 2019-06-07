CXX=CC
FLAGS= -O3  -lm -I/opt/intel/parallel_studio_xe_2017_ce/compilers_and_libraries_2017.3.191/linux/mkl/include/ -I${DATA}/opt/include -lfftw3 -lfftw3_threads -lfftw3_mpi -lboost_system -lboost_chrono -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -liomp5 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64



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
