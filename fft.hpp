#ifndef FFTW_HPP
#define FFTW_HPP

#include "complex.hpp"
#include <fftw3.h>
#include <math.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <mkl_service.h>
#include <mkl_dfti.h>
#include "fftpack.h"
#include <casacore/scimath/Mathematics/FFTPack.h>

   
class fft{
public:
  typedef int size_t;
protected:
  size_t n_dimensions;
  size_t* n;
  bool inverse;

public:
  fft(const size_t i_n_dimensions, const size_t* i_n, bool i_inverse){
    n_dimensions=i_n_dimensions;
    n=new size_t[n_dimensions];
    for(size_t i=0; i<n_dimensions; ++i){
      n[i]=i_n[i];
    }
    inverse=i_inverse;
  }
  ~fft(){
    delete[] n;
  }
  size_t get_n_dimensions() const {
    return n_dimensions;
  }
  size_t size(const size_t i) const {
    return n[i];
  }
  size_t size() const {
    size_t s=1;
    for(size_t i=0; i<n_dimensions; ++i){
      s=s*n[i];
    }
    return s;
  }
  bool is_inverse() const {
    return inverse;   
  }
  virtual size_t size_complex() const = 0;
  virtual int compute(::complex* in, ::complex* out) const {}
  virtual int compute(double* in, ::complex* out) const {}
};


class fftw : public fft {
protected:
  fftw_plan plan;
public:
  fftw(const size_t i_n_dimensions, const size_t* i_n, bool i_inverse) : fft(i_n_dimensions, i_n, i_inverse){
  }
};


class fftw_c2c : public fftw{
private:
  mutable fftw_complex *fftw_in, *fftw_out;
public:
  fftw_c2c(const size_t i_n_dimensions, const size_t* i_n_x, bool i_inverse=false) : fftw(i_n_dimensions, i_n_x, i_inverse){
    fftw_in=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size());
    fftw_out=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size());
    if(!is_inverse()){
      plan=fftw_plan_dft(n_dimensions, n, fftw_in, fftw_out,FFTW_FORWARD, FFTW_ESTIMATE);
    }else{
      plan=fftw_plan_dft(n_dimensions, n, fftw_in, fftw_out,FFTW_BACKWARD, FFTW_ESTIMATE);  
    }
  }
  ~fftw_c2c(){
    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
  }
  size_t size_complex() const {
    return size();
  }
  int compute(::complex* in, ::complex* out) const {
    size_t i;
    if(!is_inverse()){
      for(i=0; i<size(); ++i){
        fftw_in[i][0]=in[i].real();
        fftw_in[i][1]=in[i].imag();
      }
      fftw_execute(plan);
      for(i=0; i<size(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=0; i<size(); ++i){
        fftw_in[i][0]=out[i].real();
        fftw_in[i][1]=out[i].imag();
      }
      fftw_execute(plan);
      for(i=0; i<size(); ++i){
        in[i].x=fftw_out[i][0]/size();
        in[i].y=fftw_out[i][1]/size();
      }       
    }
    return 0;
  }
};


class fftw_r2c : public fftw{
private:
  mutable double* fftw_in;
  mutable fftw_complex* fftw_out;
public:
  fftw_r2c(const size_t i_n_dimensions, const size_t* i_n_x, bool i_inverse=false) : fftw(i_n_dimensions, i_n_x, i_inverse){
    fftw_in=(double*)fftw_malloc(sizeof(double)*size());
    fftw_out=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size());
    if(!is_inverse()){
      plan=fftw_plan_dft_r2c(n_dimensions, n, fftw_in, fftw_out, FFTW_ESTIMATE);
    }else{
      plan=fftw_plan_dft_c2r(n_dimensions, n, fftw_out, fftw_in, FFTW_ESTIMATE);  
    }
  }
  ~fftw_r2c(){
    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
  }
  size_t size_complex() const {
    size_t s=1;
    for(size_t i=0; i<n_dimensions-1; ++i){
      s=s*n[i];
    }
    s=s*(n[n_dimensions-1]/2+1);
    return s;
  }
  int compute(double* in, ::complex* out) const {
    size_t i;
    if(!is_inverse()){
      for(i=0; i<size(); ++i){
        fftw_in[i]=in[i];
      }
      fftw_execute(plan);
      for(i=0; i<size_complex(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=0; i<size_complex(); ++i){
        fftw_out[i][0]=out[i].x;
        fftw_out[i][1]=out[i].y;
      }
      fftw_execute(plan);
      for(i=0; i<size(); ++i){
        in[i]=fftw_in[i]/size();
      }
    }
    return 0;
  }
};


class gsl_fft : public fft {
protected:

public:
  gsl_fft(const size_t i_n, bool i_inverse) : fft(1, &i_n, i_inverse){
  }
};


class gsl_fft_c2c : public gsl_fft{
private:
  mutable double* data;
  gsl_fft_complex_wavetable* wavetable;
  gsl_fft_complex_workspace* workspace;
public:
  gsl_fft_c2c(const size_t i_n, bool i_inverse=false) : gsl_fft(i_n, i_inverse){
    data=new double[2*size()];
    wavetable=gsl_fft_complex_wavetable_alloc(n[0]);
    workspace=gsl_fft_complex_workspace_alloc(n[0]);
  }
  ~gsl_fft_c2c(){
    delete[] data;
    gsl_fft_complex_wavetable_free(wavetable);
    gsl_fft_complex_workspace_free(workspace);
  }
  size_t size_complex() const {
    return size();
  }
  int compute(::complex* in, ::complex* out) const {
    if(!is_inverse()){
      for(size_t i=0; i<n[0]; ++i){
        data[2*i]=in[i].real();
        data[2*i+1]=in[i].imag();
      }
      gsl_fft_complex_forward(data, 1, n[0], wavetable, workspace);
      for(size_t i=0; i<n[0]; ++i){
        out[i].x=data[2*i];
        out[i].y=data[2*i+1];
      }
    }else{
      for(size_t i=0; i<n[0]; ++i){
        data[2*i]=out[i].real();
        data[2*i+1]=out[i].imag();
      }
      gsl_fft_complex_backward(data, 1, n[0], wavetable, workspace);
      for(size_t i=0; i<n[0]; ++i){
        in[i].x=data[2*i]/size();
        in[i].y=data[2*i+1]/size();
      }
    }
    return 0;
  }
};

class gsl_fft_r2c : public gsl_fft{
private:
  mutable double* data;
  gsl_fft_real_wavetable* wavetable;
  gsl_fft_halfcomplex_wavetable* inverse_wavetable;
  gsl_fft_real_workspace* workspace;
  
public:
  gsl_fft_r2c(const size_t i_n, bool i_inverse=false) : gsl_fft(i_n, i_inverse){
    data=new double[2*size()];
    if(!is_inverse()){
      wavetable=gsl_fft_real_wavetable_alloc(n[0]);
    }else{
      inverse_wavetable=gsl_fft_halfcomplex_wavetable_alloc(n[0]);
    }
    workspace=gsl_fft_real_workspace_alloc(n[0]);
  }
  ~gsl_fft_r2c(){
    delete[] data;  
    if(!is_inverse()){
      gsl_fft_real_wavetable_free(wavetable);
    }else{
      gsl_fft_halfcomplex_wavetable_free(inverse_wavetable);
    }
    gsl_fft_real_workspace_free(workspace);
  }
  size_t size_complex() const {
    return size()/2+1;
  }
  int compute(double* in, ::complex* out) const {
    if(!is_inverse()){      
      for(size_t i=0; i<n[0]; ++i){
        data[i]=in[i];
      }    
      gsl_fft_real_transform(data, 1, n[0], wavetable, workspace);
      out[0].x=data[0];
      out[0].y=0;
      for(size_t i=1; i<size_complex(); ++i){
        out[i].x=data[2*i-1];
        out[i].y=data[2*i];
      }
      if(size_complex()>size()/2){
        out[size_complex()].x=data[size_complex()-1];
        out[size_complex()].y=-data[size_complex()];
      }
    }else{
      data[0]=out[0].x;
      for(size_t i=1; i<size_complex(); ++i){
        data[2*i-1]=out[i].real();
        data[2*i]=out[i].imag();
      }
      if(size_complex()>size()/2){
        data[size_complex()-1]=out[size_complex()].x;
        data[size_complex()]=-out[size_complex()].y;
      }
      gsl_fft_halfcomplex_inverse(data, 1, n[0], inverse_wavetable, workspace);
      for(size_t i=0; i<n[0]; ++i){
        in[i]=data[i];
      }
    }
      return 0;
  }
};


class mkl_fft : public fft {
public:

protected:
  DFTI_DESCRIPTOR_HANDLE descriptor_handle;
public:
  mkl_fft(const size_t i_n_dimensions, const size_t* i_n, bool i_inverse) : fft(i_n_dimensions, i_n, i_inverse){
  }
  ~mkl_fft(){
    DftiFreeDescriptor(&descriptor_handle);
  }
};


class mkl_fft_c2c : public mkl_fft{
private:
  mutable MKL_Complex16* data;
public:
  mkl_fft_c2c(const size_t i_n_dimensions, const size_t* i_n_x, bool i_inverse=false) : mkl_fft(i_n_dimensions, i_n_x, i_inverse){
    data=(MKL_Complex16*)mkl_malloc(size()*sizeof(MKL_Complex16)+2, 64);
    if(n_dimensions==1){
      DftiCreateDescriptor(&descriptor_handle, DFTI_DOUBLE, DFTI_COMPLEX, n_dimensions, (long)n[0]);
    }else{
      long nl[n_dimensions];
      for(size_t i=0; i<n_dimensions; ++i){
        nl[i]=(long)n[i];
      }
      DftiCreateDescriptor(&descriptor_handle, DFTI_DOUBLE, DFTI_COMPLEX, n_dimensions, nl);
    }
    DftiCommitDescriptor(descriptor_handle);
  }
  ~mkl_fft_c2c(){
    mkl_free(data);   
  }
  size_t size_complex() const {
    return size();
  }
  int compute(::complex* in, ::complex* out) const {
    size_t i;
    if(!is_inverse()){
      for(i=0; i<size(); ++i){
        data[i].real=in[i].real();
        data[i].imag=in[i].imag();
      }
      DftiComputeForward(descriptor_handle, data);
      for(i=0; i<size(); ++i){
        out[i].x=data[i].real;
        out[i].y=data[i].imag;
      }
    }else{
      for(i=0; i<size(); ++i){
        data[i].real=out[i].real();
        data[i].imag=out[i].imag();
      }
      DftiComputeBackward(descriptor_handle, data);
      for(i=0; i<size_complex(); ++i){
        in[i].x=data[i].real/size();
        in[i].y=data[i].imag/size();
      }
    }
    return 0;
  }
};


class mkl_fft_r2c : public mkl_fft{
private:
  double* data;
  MKL_Complex16* transform;
  long *rstrides, *cstrides;
public:
  mkl_fft_r2c(const size_t i_n_dimensions, const size_t* i_n_x, bool i_inverse=false) : mkl_fft(i_n_dimensions, i_n_x, i_inverse){
    data=(double*)mkl_malloc(size_complex()*2*sizeof(double), 64);
    transform=(MKL_Complex16*)mkl_malloc(size_complex()*sizeof(MKL_Complex16), 64);
    if(n_dimensions==1){
      DftiCreateDescriptor(&descriptor_handle, DFTI_DOUBLE, DFTI_REAL, n_dimensions, n[0]);
    }else{
      long nl[n_dimensions];
      for(size_t i=0; i<n_dimensions; ++i){
        nl[i]=(long)n[i];
      }
      DftiCreateDescriptor(&descriptor_handle, DFTI_DOUBLE, DFTI_REAL, n_dimensions, nl);
    }
    DftiSetValue(descriptor_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(descriptor_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    
    rstrides=new long[n_dimensions+1];
    cstrides=new long[n_dimensions+1];
    rstrides[n_dimensions]=1;
    cstrides[n_dimensions]=1;
    rstrides[0]=0;
    cstrides[0]=0;
    long n22;
    switch(n_dimensions){
      case 1:
        break;
      case 2:
        rstrides[1]=n[1];
        cstrides[1]=n[1]/2+1;
        break;
      case 3:
        n22=n[2]/2+1;
        rstrides[1]=2*n[1]*n22;
        cstrides[1]=n[1]*n22;
        rstrides[2]=2*n22;
        cstrides[2]=n22;
        break;
      default:
        std::cout << "The strides are computed only in 1, 2 or 3 dimensions.\n";
         break; 
    }
    if(!is_inverse()){
      DftiSetValue(descriptor_handle, DFTI_INPUT_STRIDES, rstrides);
      DftiSetValue(descriptor_handle, DFTI_OUTPUT_STRIDES, cstrides);
    }else{
      DftiSetValue(descriptor_handle, DFTI_INPUT_STRIDES, cstrides);
      DftiSetValue(descriptor_handle, DFTI_OUTPUT_STRIDES, rstrides);
    }
    DftiCommitDescriptor(descriptor_handle);
 //   std::cout << "strides\n";
 //   for(size_t i=0; i<=n_dimensions; ++i){
 //     std::cout << rstrides[i] << " " << cstrides[i] << "\n";   
 //   }
  }
  ~mkl_fft_r2c(){
    mkl_free(data);
    mkl_free(transform);
    delete[] rstrides;
    delete[] cstrides;
  }
  size_t size_complex() const {
    size_t s=1;
    for(size_t i=0; i<n_dimensions-1; ++i){
      s=s*n[i];
    }
    s=s*(n[n_dimensions-1]/2+1);
    return s;
  }
  int compute(double* in, ::complex* out) const {
    size_t i;
    if(!is_inverse()){
      for(i=0; i<size(); ++i){
        data[i]=in[i];
      }
      DftiComputeForward(descriptor_handle, data, transform);
      for(i=0; i<size_complex(); ++i){
        out[i].x=transform[i].real;
        out[i].y=transform[i].imag;
      }
    }else{
      for(i=0; i<size_complex(); ++i){
        transform[i].real=out[i].real();
        transform[i].imag=out[i].imag();
      }
      DftiComputeBackward(descriptor_handle, transform, data);
      for(i=0; i<size(); ++i){
        in[i]=data[i]/size();
      }
    }
    return 0;
  }
};


class fftpack : public fft {
protected:
    typedef casacore::DComplex casa_complex;
    typedef casacore::Double casa_double;
    mutable casa_double* w;
    mutable casacore::FFTPack* F;
     
public:
  fftpack(const size_t i_n, bool i_inverse) : fft(1, &i_n, i_inverse){
    w=new casa_double[4*size()+15];
    F=new casacore::FFTPack();
  }
  ~fftpack(){
    delete[] w;
    delete F;
  }
};


class fftpack_c2c : public fftpack{
private:
  mutable casa_complex* data; 
public:
  fftpack_c2c(const size_t i_n, bool i_inverse=false) : fftpack(i_n, i_inverse){
    data=new casa_complex[size()];
    F->cffti(size(), w);
  }
  ~fftpack_c2c(){
    delete[] data;
  }
  size_t size_complex() const {
    return size();
  }
  int compute(::complex* in, ::complex* out) const {
    if(is_inverse()){
      ::complex* buffer;
      buffer=in;
      in=out;
    }
    for(size_t i=0; i<n[0]; ++i){
      data[i]=casa_complex(in[i].real(), in[i].imag());
    }
    F->cfftf(size(), data, w);
    for(size_t i=0; i<n[0]; ++i){
      out[i].x=data[i].real();
      out[i].y=data[i].imag();
    }
    return 0;
  }
};

class fftpack_r2c : public fftpack{
private:
  mutable casa_double* data;
public:
  fftpack_r2c(const size_t i_n, bool i_inverse=false) : fftpack(i_n, i_inverse){
    data=new casa_double[size()];
    F->rffti(size(), w);
  }
  ~fftpack_r2c(){
    delete[] data;
  }
  size_t size_complex() const {
    return ceil(((double)size())/2);
  }
  int compute(double* in, ::complex* out) const {
    if(!is_inverse()){
      for(size_t i=0; i<n[0]; ++i){
        data[i]=in[i];
      }
      F->rfftf(size(), data, w);
      out[0].x=data[0];
      out[0].y=0;
      for(size_t i=1; i<size_complex(); ++i){
        out[i].x=data[2*i-1];
        out[i].y=data[2*i];
      }
      if(size_complex()>size()/2){
        out[size_complex()].x=data[size_complex()-1];
        out[size_complex()].y=-data[size_complex()];
      }
    }else{
      data[0]=out[0].x;
      for(size_t i=1; i<size_complex(); ++i){
        data[2*i-1]=out[i].real();
        data[2*i]=out[i].imag();
      }
      if(size_complex()>size()/2){
        data[size_complex()-1]=out[size_complex()].x;
        data[size_complex()]=-out[size_complex()].y;
      }
      F->rfftb(size(), data, w);
      for(size_t i=0; i<n[0]; ++i){
        in[i]=data[i]/size();
      }
    }
    return 0;
  }
};


#endif
