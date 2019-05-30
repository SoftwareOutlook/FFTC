#include "complex.hpp"
#include "signal.hpp"
#include "fft.hpp"
#include "fftmpi.hpp"
#include <iostream>
#include "stopwatch.hpp"
#include <omp.h>
#include "multiarray.hpp"
#include <vector>
#include <mpi.h>
using namespace std;

void display(const string& s){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0){
#pragma omp parallel
#pragma omp master
    std::cout << s;
  }
}

int main(int argc, char** argv){

  MPI_Init(&argc, &argv);
    
  if(argc<5){
    display( "Please supply 3 dimensions and a number of coils\n");
    return -1;
  }



#pragma omp parallel
#pragma omp master
  display("N threads: "+std::to_string(omp_get_num_threads())+"\n\n");

  fftw_mpi_init();
  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
  
  mkl_set_num_threads_local(omp_get_max_threads());
  
  
  fft_mpi::size_t i, j, k[3], n_x[3], dim[3], n_dimensions, n_coils;
  double a=0.25;
  double b=a/2;
  double s, x[3], l_x[3]={1, 1, 1}, error;
  for(i=0; i<3; ++i){
    dim[i]=atoi(*(argv+i+1));
  }
  n_coils=atoi(*(argv+4));
  
  
  stopwatch sw;


  // 1D
  
  n_dimensions=1;
  n_x[0]=dim[0];
  
  // R <-> HC
  
  {
    // Dimensions
    
    display("Dimensions: ");
    for(i=0; i<n_dimensions; ++i){
      display(std::to_string(n_x[i])+" ");
    }
    display("\n");
    display("N coils: "+std::to_string(n_coils)+"\n");
    display("\n");
    display("  R <-> HC\n");
    display("\n");
    
    // Signal
    multiarray<double> sig({n_x[0]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      s=signal(n_dimensions, l_x, a, b, x);
      sig(k[0])=s;
    }
    
    // Coils
    multiarray<double> coil({n_x[0]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      coil(k[0])=1./n_coils;
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    // Multiplied signals
    std::vector<multiarray<double>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<double>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0]/2+2})); // +2 rather than +1 because of GSL
      inverse_transforms.push_back(multiarray<double>({n_x[0]}));
    }
    
    
    // FFTW
    display( "    FFTW\n");
    display( "      Direct\n");

    sw.start();
    fftw_mpi_r2c fw(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DFFTWRINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DFFTWR "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
 
    display("      Inverse\n");

    sw.start();
    fftw_mpi_r2c fwi(n_dimensions, n_x, true);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DFFTWRINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DFFTWRINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    
    
    
    /*    
    display("    MKL\n");
    display("      Direct\n");
    
    sw.start();
    mkl_fft_r2c mkl(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DMKLRINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DMKLR "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    display("      Inverse\n");
    sw.start();
    mkl_fft_r2c mkli(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DMKLRINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DMKLRINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    */
    
  }
  
  
    // C <-> C
  
  {
    // Dimensions
    display( "  C <-> C\n");
    display( "\n");
    
    // Signal
    multiarray<::complex> sig({n_x[0]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      s=signal(n_dimensions, l_x, a, b, x);
      sig(k[0])=::complex(s, s);
    }
    
    // Coils
    multiarray<double> coil({n_x[0]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      coil(k[0])=1./n_coils;
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    // Multiplied signals
    std::vector<multiarray<::complex>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<::complex>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0]})); 
      inverse_transforms.push_back(multiarray<::complex>({n_x[0]}));
    }
    
    
    // FFTW
    display( "    FFTW\n");
    display( "      Direct\n");

    sw.start();
    fftw_mpi_c2c fw(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DFFTWCINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DFFTWC "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    
    display("      Inverse\n");

    sw.start();
    fftw_mpi_c2c fwi(n_dimensions, n_x, true);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DFFTWCINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DFFTWCINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n"); 
    
    
    /*
    // MKL
    display("    MKL\n");
    display("      Direct\n");
    
    sw.start();
    mkl_fft_c2c mkl(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DMKLCINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DMKLC "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    
    display("      Inverse\n");

    sw.start();
    mkl_fft_c2c mkli(n_dimensions, n_x, true);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DMKLCINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      gsli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 1DMKLCINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");    
    */     
    
  }
  
   
  
  // 2D
  n_dimensions=2;
  n_x[0]=dim[0];
  n_x[1]=dim[1];
  
  
  // R <-> HC
  
  {
    // Dimensions
    
    display("Dimensions: ");
    for(i=0; i<n_dimensions; ++i){
      display(std::to_string(n_x[i])+" ");
    }
    display("\n");
    display("N coils: "+std::to_string(n_coils)+"\n");
    display("\n");
    display("  R <-> HC\n");
    display("\n");
    
    // Signal
    multiarray<double> sig({n_x[0], n_x[1]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){
        x[1]=((double)k[1])/n_x[1];  
        s=signal(n_dimensions, l_x, a, b, x);
        sig(k[0], k[1])=s;
      }
    }
    
    // Coils
    multiarray<double> coil({n_x[0], n_x[1]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){  
        coil(k[0], k[1])=1./n_coils;
      }
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    // Multiplied signals
    std::vector<multiarray<double>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<double>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]/2+1}));
      inverse_transforms.push_back(multiarray<double>({n_x[0], n_x[1]}));
    }
    
    
    // FFTW
    display("    FFTW\n");
    display("      Direct\n");

    sw.start();
    fftw_mpi_r2c fw(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DFFTWRINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DFFTWR "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    
    display("      Inverse\n");

    sw.start();
    fftw_mpi_r2c fwi(n_dimensions, n_x, true);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DFFTWRINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DFFTWRINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    
    /*  
    // MKL
    display("    MKL\n");
    display("      Direct\n");

    sw.start();
    mkl_fft_r2c mkl(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DMKLRINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DMKLR "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
     
    display("      Inverse\n");

    sw.start();
    mkl_fft_r2c mkli(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DMKLRINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DMKLR "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");

    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    */
  }  
  
  
  // C <-> C
  
  {
    display( "  C <-> C\n");
    display( "\n");
    
    // Signal
    multiarray<::complex> sig({n_x[0], n_x[1]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){
        x[1]=((double)k[1])/n_x[1];  
        s=signal(n_dimensions, l_x, a, b, x);
        sig(k[0], k[1])=::complex(s, s);
      }
    }
    
    // Coils
    multiarray<double> coil({n_x[0], n_x[1]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){  
        coil(k[0], k[1])=1./n_coils;
      }
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    // Multiplied signals
    std::vector<multiarray<::complex>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<::complex>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]}));
      inverse_transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]}));
    }
    
    
    // FFTW
    display("    FFTW\n");
    display("      Direct\n");

    sw.start();
    fftw_mpi_c2c fw(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DFFTWCINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DFFTWC "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    
    display("      Inverse\n");

    sw.start();
    fftw_mpi_c2c fwi(n_dimensions, n_x, true);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DFFTWCINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DFFTWCINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    
    /*
    // MKL
    display("    MKL\n");
    display("      Direct\n");

    sw.start();
    mkl_fft_c2c mkl(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DMKLCINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DMKLC "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    display("      Inverse\n");

    sw.start();
    mkl_fft_c2c mkli(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DMKLCINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 2DMKLCINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    */
  }  
  
  
  
  // 3D
  n_dimensions=3;
  n_x[0]=dim[0];
  n_x[1]=dim[1];
  n_x[2]=dim[2];
  
  
  // R <-> HC
  
  {
    // Dimensions
    
    display( "Dimensions: ");
    for(i=0; i<n_dimensions; ++i){
      display(std::to_string(n_x[i])+" ");
    }
    display("\n");
    display("N coils: "+std::to_string(n_coils)+"\n");
    display("\n");
    display("  R <-> HC\n");
    display("\n");
    
    // Signal
    multiarray<double> sig({n_x[0], n_x[1], n_x[2]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){
        x[1]=((double)k[1])/n_x[1];
        for(k[2]=0; k[2]<n_x[2]; ++k[2]){
          x[2]=((double)k[2])/n_x[2];
          s=signal(n_dimensions, l_x, a, b, x);
          sig(k[0], k[1], k[2])=s;
        }
      }
    }
    
    // Coils
    multiarray<double> coil({n_x[0], n_x[1], n_x[2]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){  
        for(k[2]=0; k[2]<n_x[2]; ++k[2]){   
          coil(k[0], k[1], k[2])=1./n_coils;
        }
      }
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    // Multiplied signals
    std::vector<multiarray<double>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<double>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1], n_x[2]/2+1}));
      inverse_transforms.push_back(multiarray<double>({n_x[0], n_x[1], n_x[2]}));
    }
    
    
    // FFTW
    display("    FFTW\n");
    display("      Direct\n");

    sw.start();
    fftw_mpi_r2c fw(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DFFTWRINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n"); 
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DFFTWR "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n"); 
    
    display("      Inverse\n");

    sw.start();
    fftw_mpi_r2c fwi(n_dimensions, n_x, true);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DFFTWRINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n"); 
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DFFTWRINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n"); 
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    
    /*
    // MKL
    display("    MKL\n");
    display("      Direct\n");
    display("@ 3DMKLRINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");  
    mkl_fft_r2c mkl(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DMKLR "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");  

    display("      Inverse\n");

    sw.start();
    mkl_fft_r2c mkli(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DMKLRINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n"); 
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DMKLRINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n"); 
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    */
  }  
  
  
  // C <-> C
  
  {
    display("  C <-> C\n");
    display("\n");
    
    // Signal
    multiarray<::complex> sig({n_x[0], n_x[1], n_x[2]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){
        x[1]=((double)k[1])/n_x[1];
        for(k[2]=0; k[2]<n_x[2]; ++k[2]){
          x[2]=((double)k[2])/n_x[2];
          s=signal(n_dimensions, l_x, a, b, x);
          sig(k[0], k[1], k[2])=::complex(s, s);
        }
      }
    }
    
    // Coils
    multiarray<double> coil({n_x[0], n_x[1], n_x[2]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){  
        for(k[2]=0; k[2]<n_x[2]; ++k[2]){   
          coil(k[0], k[1], k[2])=1./n_coils;
        }
      }
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    // Multiplied signals
    std::vector<multiarray<::complex>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<::complex>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1], n_x[2]}));
      inverse_transforms.push_back(multiarray<::complex>({n_x[0], n_x[1], n_x[2]}));
    }
    
   
    // FFTW
    display("    FFTW\n");
    display("      Direct\n");

    sw.start();
    fftw_mpi_c2c fw(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DFFTWCINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DFFTWC "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    
    display("      Inverse\n");

    sw.start();
    fftw_mpi_c2c fwi(n_dimensions, n_x, true);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DFFTWCINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DFFTWCINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    
    /*
    // MKL
    display("    MKL\n");
    display("      Direct\n");

    sw.start();
    mkl_fft_c2c mkl(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DMKLCINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DMKLC "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    display("      Inverse\n");

    sw.start();
    mkl_fft_c2c mkli(n_dimensions, n_x);
    sw.stop();
    display("        Init. time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DMKLCINVINIT "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    display("        Time:  "+std::to_string(sw.get())+" s\n");
    display("@ 3DMKLCINV "+std::to_string(n_x[0])+" "+std::to_string(sw.get())+"\n");
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    display("      Error: "+std::to_string(error)+"\n");
    display("\n");
    */
  }  
   
   
  fftw_cleanup_threads();

  MPI_Finalize();

  return 0;
  
}
