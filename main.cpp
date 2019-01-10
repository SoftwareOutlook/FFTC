#include "complex.hpp"
#include "signal.hpp"
#include "fft.hpp"
#include <iostream>
#include "stopwatch.hpp"
#include <omp.h>
#include "multiarray.hpp"
using namespace std;


int main(int argc, char** argv){

   // cfftf1(
    
  if(argc<4){
    std::cout << "Please supply 3 dimensions\n";
    return -1;
  }


#pragma omp parallel
#pragma omp master
  std::cout << "N threads: " << omp_get_num_threads() << "\n\n";


  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
  
  mkl_set_num_threads_local(omp_get_max_threads());
  
  
  int i, j, k[3], n_x[3], dim[3], n_dimensions;
  double a=0.25;
  double b=a/2;
  double s, x[3], l_x[3]={1, 1, 1};
  for(i=0; i<3; ++i){
    dim[i]=atoi(*(argv+i+1));
  }
  unsigned long long n_max_elements=dim[0]*dim[1]*dim[2]+1;
  double* signal_r=new double[n_max_elements];
  ::complex* signal_c=new ::complex[n_max_elements];
  ::complex* transform=new ::complex[n_max_elements];
  stopwatch sw;


  // 1D

  // Signal
  n_dimensions=1;
  n_x[0]=dim[0]*dim[1];
  std::cout << "Dimensions: ";
  for(i=0; i<n_dimensions; ++i){
    std::cout << n_x[i] << " ";
  }
  std::cout << "\n";
  for(k[0]=0; k[0]<n_x[0]; ++k[0]){
    x[0]=((double)k[0])/n_x[0];
    s=signal(1, l_x, a, b, x);
    signal_r[k[0]]=s;
    signal_c[k[0]]=::complex(s, s);
  }


  // FFTW
  std::cout << "  FFTW\n";

  // r->c
  std::cout << "    r->c\n";
  fftw_r2c fw_1d_r2c(n_dimensions, n_x);
  sw.start();
  fw_1d_r2c.compute(signal_r, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(fw_1d_r2c, signal_r, transform) << "\n\n";


  // c->c
  std::cout << "    c->c\n";
  fftw_c2c fw_1d_c2c(n_dimensions, n_x);
  sw.start();
  fw_1d_c2c.compute(signal_c, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(fw_1d_c2c, signal_c, transform) << "\n\n";

  // GSL
  std::cout << "  GSL\n";

  // r->c
  std::cout << "    r->c\n";
  gsl_fft_r2c gsl_r2c(n_x[0]);
  sw.start();
  gsl_r2c.compute(signal_r, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(gsl_r2c, signal_r, transform) << "\n\n";

  // c->c
  std::cout << "    c->c\n";
  gsl_fft_c2c gsl_c2c(n_x[0]);
  sw.start();
  gsl_c2c.compute(signal_c, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(gsl_c2c, signal_c, transform) << "\n\n";
 

  // MKL
  std::cout << "  MKL\n";

  // r->c
  std::cout << "    r->c\n";
  mkl_fft_r2c mkl_1d_r2c(n_dimensions, n_x);
  sw.start();
  mkl_1d_r2c.compute(signal_r, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(mkl_1d_r2c, signal_r, transform) << "\n\n";

  // c->c
  std::cout << "    c->c\n";
  mkl_fft_c2c mkl_1d_c2c(n_dimensions, n_x);
  sw.start();
  mkl_1d_c2c.compute(signal_c, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(mkl_1d_c2c, signal_c, transform) << "\n\n";

  
  // FFTPACK
  std::cout << "  FFTPACK\n";

  // r->c
    std::cout << "    r->c\n";
    fftpack_r2c fp_1d_r2c(n_x[0]);
    sw.start();
    fp_1d_r2c.compute(signal_r, transform);
    sw.stop();
    std::cout << "      Time:  " << sw.get() << " s\n";
    std::cout << "      Error: " << error(fp_1d_r2c, signal_r, transform) << "\n\n";

  // c->c
  std::cout << "    c->c\n";
  fftpack_c2c fp_1d_c2c(n_x[0]);
  sw.start();
  fp_1d_c2c.compute(signal_c, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(fp_1d_c2c, signal_c, transform) << "\n\n";


  // 2D

  // Signal
  n_dimensions=2;
  n_x[0]=dim[0];
  n_x[1]=dim[1];
  std::cout << "Dimensions: ";
  for(i=0; i<n_dimensions; ++i){
    std::cout << n_x[i] << " ";
  }
  std::cout << "\n";
  for(k[0]=0; k[0]<n_x[0]; ++k[0]){
    x[0]=((double)k[0])/n_x[0];
    for(k[1]=0; k[1]<n_x[1]; ++k[1]){
      x[1]=((double)k[1])/n_x[1];
      s=signal(n_dimensions, l_x, a, b, x);
      signal_r[k[0]*n_x[1]+k[1]]=s;
      signal_c[k[0]*n_x[1]+k[1]]=::complex(s, s);
    }
  }


  // FFTW
  std::cout << "  FFTW\n";

  // r->c
  std::cout << "    r->c\n";
  fftw_r2c fw_2d_r2c(n_dimensions, n_x);
  sw.start();
  fw_2d_r2c.compute(signal_r, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(fw_2d_r2c, signal_r, transform) << "\n\n";

  // c->c
  std::cout << "    c->c\n";
  fftw_c2c fw_2d_c2c(n_dimensions, n_x);
  sw.start();
  fw_2d_c2c.compute(signal_c, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(fw_2d_c2c, signal_c, transform) << "\n\n";

  // MKL
  std::cout << "  MKL\n";

  // r->c
  std::cout << "    r->c\n";
  mkl_fft_r2c mkl_2d_r2c(n_dimensions, n_x);
  sw.start();
  mkl_2d_r2c.compute(signal_r, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(mkl_2d_r2c, signal_r, transform) << "\n\n";
  
  // c->c
  std::cout << "    c->c\n";
  mkl_fft_c2c mkl_2d_c2c(n_dimensions, n_x);
  sw.start();
  mkl_2d_c2c.compute(signal_c, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(mkl_2d_c2c, signal_c, transform) << "\n\n";




  // 3D

  // Signal
  n_dimensions=3;
  n_x[0]=dim[0];
  n_x[1]=dim[1];
  n_x[2]=dim[2];
  std::cout << "Dimensions: ";
  for(i=0; i<n_dimensions; ++i){
    std::cout << n_x[i] << " ";
  }
  std::cout << "\n";
  for(k[0]=0; k[0]<n_x[0]; ++k[0]){
    x[0]=((double)k[0])/n_x[0];
    for(k[1]=0; k[1]<n_x[1]; ++k[1]){
      x[1]=((double)k[1])/n_x[1];
      for(k[2]=0; k[2]<n_x[2]; ++k[2]){
        x[2]=((double)k[2])/n_x[2];
        s=signal(n_dimensions, l_x, a, b, x);
        signal_r[k[0]*n_x[1]*n_x[2]+k[1]*n_x[2]+k[2]]=s;
        signal_c[k[0]*n_x[1]*n_x[2]+k[1]*n_x[2]+k[2]]=::complex(s, s);
      }
    }
  }
  

  // FFTW
  std::cout << "  FFTW\n";

  // r->c
  std::cout << "    r->c\n";
  fftw_r2c fw_3d_r2c(n_dimensions, n_x);
  sw.start();
  fw_3d_r2c.compute(signal_r, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(fw_3d_r2c, signal_r, transform) << "\n\n";

  // c->c
  std::cout << "    c->c\n";
  fftw_c2c fw_3d_c2c(n_dimensions, n_x);
  sw.start();
  fw_3d_c2c.compute(signal_c, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(fw_3d_c2c, signal_c, transform) << "\n\n";

  
  // MKL
  std::cout << "  MKL\n";

  // r->c
  std::cout << "    r->c\n";
  mkl_fft_r2c mkl_3d_r2c(n_dimensions, n_x);
  sw.start();
  mkl_3d_r2c.compute(signal_r, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(mkl_3d_r2c, signal_r, transform) << "\n\n";

  // c->c
  std::cout << "    c->c\n";
  mkl_fft_c2c mkl_3d_c2c(n_dimensions, n_x);
  sw.start();
  mkl_3d_c2c.compute(signal_c, transform);
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  std::cout << "      Error: " << error(mkl_3d_c2c, signal_c, transform) << "\n\n\n";
  
  
  /*
  // CCP PET/MR
  std::cout << "CCP PET/MR\n";
  unsigned int n_coils=32;
  std::cout << "N images: " << n_x[2] << "\n";
  std::cout << "Images size: " << n_x[0] << " x " << n_x[1] << "\n";
  std::cout << "N coils:     " << n_coils << "\n";
  
  // Data
  cube<::complex> slices(n_x[0], n_x[1], n_x[2]);
  cube<::complex> transforms(n_x[0], n_x[1], n_x[2]);
  for(k[0]=0; k[0]<n_x[0]; ++k[0]){
    for(k[1]=0; k[1]<n_x[1]; ++k[1]){
      for(k[2]=0; k[2]<n_x[2]; ++k[2]){  
        slices[k[0]][k[1]][k[2]]=signal_c[k[0]*n_x[1]+k[1]]/n_coils;     
      }
    }
  }
  std::cout << "  FFTW\n";
  sw.start();
  for(i=0; i<n_x[2]; ++i){
    for(j=0; j<n_coils; ++j){
      fw_2d_c2c.compute(slices.get_pointer_to_slice(i), transforms.get_pointer_to_slice(i));
    }
  }
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  double e;
  e=0;
  for(i=0; i<n_x[2]; ++i){
    e=e+error(fw_2d_c2c, slices.get_pointer_to_slice(i), transforms.get_pointer_to_slice(i));
  }
  std::cout << "      Error: " << e << "\n\n";
  
  std::cout << "  MKL\n";
  sw.start();
  for(i=0; i<n_x[2]; ++i){
    for(j=0; j<n_coils; ++j){  
      mkl_2d_c2c.compute(slices.get_pointer_to_slice(i), transforms.get_pointer_to_slice(i));
    }
  }
  sw.stop();
  std::cout << "      Time:  " << sw.get() << " s\n";
  e=0;
  for(i=0; i<n_x[2]; ++i){
    e=e+error(mkl_2d_c2c, slices.get_pointer_to_slice(i), transforms.get_pointer_to_slice(i));
  }
  std::cout << "      Error: " << e << "\n\n";
*/  
  delete[] signal_r;
  delete[] signal_c;
  delete[] transform;
  
  
   
   
  fftw_cleanup_threads();

  return 0;

}
