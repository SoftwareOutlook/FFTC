module load gcc/7.2.0 intel/17.0.3.191

export OPT=${DATA}/opt
export C_INCLUDE_PATH=${OPT}/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${OPT}/include:${CPLUS_INCLUDE_PATH}
export LD_LIBRARY_PATH=${OPT}/lib:${LD_LIBRARY_PATH}
export PATH=${OPT}/lib:${PATH}