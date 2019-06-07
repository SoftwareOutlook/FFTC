#include <mpi.h>
#include <mkl_service.h>
#include <mkl_dfti.h>
#include <mkl_cdft.h>
#include "stopwatch.hpp"
#include "complex.hpp"
#include <omp.h>
#include <iostream>


int main(int argc, char** argv){

  MPI_Init(&argc, &argv);

  long n_dimensions=2;
  long n[2]={16, 16};
  long allocated_size, n_x_local, offset;

  DFTI_DESCRIPTOR_DM_HANDLE descriptor_handle;
  MKL_Complex16* data;

  DftiCreateDescriptorDM(MPI_COMM_WORLD, &descriptor_handle, DFTI_DOUBLE, DFTI_COMPLEX, n_dimensions, allocated_size);

  DftiGetValueDM(descriptor_handle, CDFT_LOCAL_SIZE,&allocated_size);

  data=(MKL_Complex16*)mkl_malloc(allocated_size*sizeof(MKL_Complex16)+2, 64);
  DftiGetValueDM(descriptor_handle,CDFT_LOCAL_NX, &n_x_local);
  DftiGetValueDM(descriptor_handle,CDFT_LOCAL_X_START,&offset);
DftiCommitDescriptorDM(descriptor_handle);

DftiComputeForwardDM(descriptor_handle, data,data);

DftiFreeDescriptorDM(&descriptor_handle);

 mkl_free(data);
MPI_Finalize();
  /*
  // Initialisation
    
  if(argc<4){
    return -1;   
  }
  
  MPI_Init(&argc, &argv);
  int i_process, n_processes, n_threads;
  MPI_Comm_rank(MPI_COMM_WORLD, &i_process);
  MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
#pragma omp parallel
#pragma omp master
  n_threads=omp_get_max_threads();

  if(i_process==0) std::cout << "N processes: " << n_processes << "\n";
  if(i_process==0) std::cout << "N threads: " << n_processes << "\n\n";
  
  fftw_mpi_init();
  fftw_init_threads();
  fftw_plan_with_nthreads(n_threads);
  
  long i, n[3], n_local[3], k_local[3], k_global[3];
  for(i=0; i<3; ++i){
    n[i]=atoi(*(argv+i+1));
  }
  
  long n_x_local, offset, local_size;
  long n_x_local_input, n_x_local_output, offset_input, offset_output;
  double *in_r;
  fftw_complex *in_c, *out_c;
  double *signal_r;
  complex *signal_c, *transform_c;
  fftw_plan plan;
  stopwatch sw;

  
  // 1D
  if(i_process==0) std::cout << "1D\n";
  
  //   C -> C
  if(i_process==0) std::cout << "  C -> C\n";
  
  local_size=2*fftw_mpi_local_size_1d(n[0], MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE, &n_x_local_input, &offset_input, &n_x_local_output, &offset_output);
  in_c=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*local_size);
  out_c=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*local_size);
  signal_c=new complex[local_size];
  transform_c=new complex[local_size];

  //     Signal
  n_local[0]=n_x_local_input;
  for(k_local[0]=0; k_local[0]<n_local[0]; ++k_local[0]){
    k_global[0]=k_local[0]+offset_input;
    if(k_global[0]<n[0]/2){
        signal_c[k_local[0]].x=1;
        signal_c[k_local[0]].y=1;
    }else{
       signal_c[k_local[0]].x=0;
       signal_c[k_local[0]].y=0;     
    }
  }

  //     Initialisation
  sw.start();
  plan=fftw_mpi_plan_dft(1, n, in_c, out_c, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
  sw.stop();
  if(i_process==0) std::cout << "    Initialisation: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @ FFTW1DCINIT " << n_processes << " " << sw.get() << "\n";
  
  //     Transform
  sw.start();
  for(i=0; i<n_local[0]; ++i){
    in_c[i][0]=signal_c[i].real();
    in_c[i][1]=signal_c[i].imag();
  }
  fftw_execute(plan);
  for(i=0; i<n_x_local_output; ++i){
    transform_c[i].x=out_c[i][0];
    transform_c[i].y=out_c[i][1];
  }
  sw.stop();
  if(i_process==0) std::cout << "    Execution: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @ FFTW1DCEXEC " << n_processes  << " " << sw.get() << "\n";
  
  fftw_destroy_plan(plan);
  fftw_free(in_c);
  fftw_free(out_c);
  delete[] signal_c;
  delete[] transform_c;
  
 
  
  // 2D
  if(i_process==0) std::cout << "2D\n";
  local_size=2*fftw_mpi_local_size(2, n, MPI_COMM_WORLD, &n_x_local, &offset);


  //   R -> HC
  if(i_process==0) std::cout << "  R -> HC\n";
  
  in_r=(double*)fftw_malloc(sizeof(double)*local_size);
  out_c=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*local_size);
  signal_r=new double[local_size];
  transform_c=new complex[local_size];

  //     Signal
  n_local[0]=n_x_local_input;
  n_local[1]=n[1];
  for(k_local[0]=0; k_local[0]<n_local[0]; ++k_local[0]){
    k_global[0]=k_local[0]+offset_input;
    for(k_local[1]=0; k_local[1]<n_local[1]; ++k_local[1]){
      k_global[1]=k_local[1];
      if(k_global[0]<n[0]/2 && k_global[1]<n[1]/2){
        signal_r[k_local[0]*n_local[1]+k_local[1]]=1;
      }else{
        signal_r[k_local[0]*n_local[1]+k_local[1]]=0;
      }
    }
  }

  //     Initialisation
  sw.start();
  plan=fftw_mpi_plan_dft_r2c(2, n, in_r, out_c, MPI_COMM_WORLD, FFTW_ESTIMATE);
  sw.stop();
  if(i_process==0) std::cout << "    Initialisation: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @ FFTW2DRINIT " << n_processes  << " " << sw.get() << "\n";
  
  //     Transform
  sw.start();
  for(i=0; i<n_local[0]*n_local[1]; ++i){
    in_r[i]=signal_r[i];
  }
  fftw_execute(plan);
  for(i=0; i<ceil(n_local[0]*n_local[1]/2); ++i){
    transform_c[i].x=out_c[i][0];
    transform_c[i].x=out_c[i][1];
  }
  sw.stop();
  if(i_process==0) std::cout << "    Execution: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @ FFTW2DREXEC " << n_processes  << " " << sw.get() << "\n";

  fftw_destroy_plan(plan);
  fftw_free(in_r);
  fftw_free(out_c);
  delete[] signal_r;
  delete[] transform_c;
 
  
  //   C -> C
  if(i_process==0) std::cout << "  C-> C\n";
  
  in_c=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*local_size);
  out_c=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*local_size);
  signal_c=new complex[local_size];
  transform_c=new complex[local_size];

 //     Signal
  n_local[0]=n_x_local_input;
  n_local[1]=n[1];
  n_local[2]=n[2]; 
  for(k_local[0]=0; k_local[0]<n_local[0]; ++k_local[0]){
    k_global[0]=k_local[0]+offset_input;
    for(k_local[1]=0; k_local[1]<n_local[1]; ++k_local[1]){
      k_global[1]=k_local[1];
      if(k_global[0]<n[0]/2 && k_global[1]<n[1]/2){
	signal_c[k_local[0]*n_local[1]+k_local[1]].x=1;
	signal_c[k_local[0]*n_local[1]+k_local[1]].y=1;
      }else{
	signal_c[k_local[0]*n_local[1]+k_local[1]].x=0;
	signal_c[k_local[0]*n_local[1]+k_local[1]].y=0;
      }
    }
   }
   
  //     Initialisation
  sw.start();
  plan=fftw_mpi_plan_dft(2, n, in_c, out_c, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
  sw.stop();
  if(i_process==0) std::cout << "    Initialisation: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @ FFTW2DCINIT " << n_processes  << " " << sw.get() << "\n";

  //     Transform
  sw.start();
  for(i=0; i<n_local[0]*n_local[1]; ++i){
    in_c[i][0]=signal_c[i].real();
    in_c[i][1]=signal_c[i].imag();
  }
  fftw_execute(plan);
  for(i=0; i<n_local[0]*n_local[1]; ++i){
    transform_c[i].x=out_c[i][0];
    transform_c[i].x=out_c[i][1];
  }
  sw.stop();
  if(i_process==0) std::cout << "    Execution: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @ FFTW2DCEXEC " << n_processes  << " " << sw.get() << "\n";
  
  fftw_destroy_plan(plan);
  fftw_free(in_c);
  fftw_free(out_c);
  delete[] signal_c;
  delete[] transform_c;

  
 
  // 3D
  if(i_process==0) std::cout << "3D\n";
  local_size=2*fftw_mpi_local_size(3, n, MPI_COMM_WORLD, &n_x_local, &offset);

  
  //   R -> HC
  if(i_process==0) std::cout << "  R -> HC\n";
  
  in_r=(double*)fftw_malloc(sizeof(double)*local_size);
  out_c=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*local_size);
  signal_r=new double[local_size];
  transform_c=new complex[local_size];

 //     Signal
  n_local[0]=n_x_local_input;
  n_local[1]=n[1];
  n_local[2]=n[2];
  for(k_local[0]=0; k_local[0]<n_local[0]; ++k_local[0]){
    k_global[0]=k_local[0]+offset_input;
    for(k_local[1]=0; k_local[1]<n_local[1]; ++k_local[1]){
      k_global[1]=k_local[1];
      for(k_local[2]=0; k_local[2]<n_local[2]; ++k_local[2]){
        k_global[2]=k_local[2];
        if(k_global[0]<n[0]/2 && k_global[1]<n[1]/2 && k_global[2]<n[2]/2){
	  signal_r[k_local[0]*n_local[1]*n_local[2]+k_local[1]*n_local[2]+k_local[3]]=1;
        }else{
	  signal_r[k_local[0]*n_local[1]*n_local[2]+k_local[1]*n_local[2]+k_local[3]]=0;         
	}
      }
    }
  }
  
  //     Initialisation
  sw.start();
  plan=fftw_mpi_plan_dft_r2c(3, n, in_r, out_c, MPI_COMM_WORLD, FFTW_ESTIMATE);
  sw.stop();
  if(i_process==0) std::cout << "    Initialisation: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @FFTW3DRINIT " << n_processes  << " " << sw.get() << "\n";
   
  
  //     Transform
  sw.start();
  for(i=0; i<n_local[0]*n_local[1]*n_local[2]; ++i){
    in_r[i]=signal_r[i];
  }
  fftw_execute(plan);
  for(i=0; i<ceil(n_local[0]*n_local[1]*n_local[2]/2); ++i){
    transform_c[i].x=out_c[i][0];
    transform_c[i].x=out_c[i][1];
  }
  sw.stop();
  if(i_process==0) std::cout << "    Execution: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @ FFTW3DREXEC " << n_processes  << " " << sw.get() << "\n"; 
  
  fftw_destroy_plan(plan); 
  fftw_free(in_r);
  fftw_free(out_c);
  delete[] signal_r;
  delete[] transform_c;
 
  
  //   C -> C
  if(i_process==0) std::cout << "  C -> C\n";
  
  in_c=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*local_size);
  out_c=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*local_size);
  signal_c=new complex[local_size];
  transform_c=new complex[local_size];

  //     Signal
  n_local[0]=n_x_local_input;
  n_local[1]=n[1];
  n_local[2]=n[2];
  for(k_local[0]=0; k_local[0]<n_local[0]; ++k_local[0]){
    k_global[0]=k_local[0]+offset_input;
    for(k_local[1]=0; k_local[1]<n_local[1]; ++k_local[1]){
      k_global[1]=k_local[1];
      for(k_local[2]=0; k_local[2]<n_local[2]; ++k_local[2]){
        k_global[2]=k_local[2];
        if(k_global[0]<n[0]/2 && k_global[1]<n[1]/2 && k_global[2]<n[2]/2){
	   signal_c[k_local[0]*n_local[1]*n_local[2]+k_local[1]*n_local[2]+k_local[3]].x=1;
	   signal_c[k_local[0]*n_local[1]*n_local[2]+k_local[1]*n_local[2]+k_local[3]].y=1;
        }else{
	   signal_c[k_local[0]*n_local[1]*n_local[2]+k_local[1]*n_local[2]+k_local[3]].x=0;
	   signal_c[k_local[0]*n_local[1]*n_local[2]+k_local[1]*n_local[2]+k_local[3]].y=0;         
	}
      }
    }
  }
  
  //      Initialisation
  sw.start();
  plan=fftw_mpi_plan_dft(3, n, in_c, out_c, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
  sw.stop();
  if(i_process==0) std::cout << "    Initialisation: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @ FFTW3DCINIT " << n_processes  << " " << sw.get() << "\n";

  //     Transform 
  sw.start();
  for(i=0; i<n_local[0]*n_local[1]*n_local[2]; ++i){
    in_c[i][0]=signal_c[i].real();
    in_c[i][1]=signal_c[i].imag();
  }
  fftw_execute(plan);
  for(i=0; i<n_local[0]*n_local[1]*n_local[2]; ++i){
    transform_c[i].x=out_c[i][0];
    transform_c[i].x=out_c[i][1];
  }
  sw.stop();
  if(i_process==0) std::cout << "    Execution: " << n_processes << " " << sw.get() << "\n";
  if(i_process==0) std::cout << "    @ FFTW3DCEXEC " << n_processes  << " " << sw.get() << "\n"; 
   
  fftw_destroy_plan(plan); 
  fftw_free(in_c);
  fftw_free(out_c);
  delete[] signal_c;
  delete[] transform_c;
  
  
  
  // Clean up
  MPI_Finalize();
  */  
  return 0;
}
