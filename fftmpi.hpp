#ifndef FFTMPI_HPP
#define FFTMPI_HPP

#include "complex.hpp"
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <mkl_service.h>
#include <mkl_dfti.h>




class fft_mpi{
public:
  typedef long size_t;
protected:
  size_t n_dimensions;
  size_t* n;
  bool inverse;
  size_t n_x_local, offset, n_x_local_output, offset_output;
  int i_process, n_processes;

public:
  fft_mpi(const size_t i_n_dimensions, const size_t* i_n, bool i_inverse){
    n_dimensions=i_n_dimensions;
    n=new size_t[n_dimensions];
    for(size_t i=0; i<n_dimensions; ++i){
      n[i]=i_n[i];
    }
    inverse=i_inverse;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_process);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
  }
  ~fft_mpi(){
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
  size_t size_local() const {
    size_t s=n_x_local;
    for(size_t i=1; i<n_dimensions; ++i){
      s=s*n[i];
    }
    return s;
  }
  size_t size_local_output() const {
    size_t s=n_x_local_output;
    for(size_t i=1; i<n_dimensions; ++i){
      s=s*n[i];
    }
    return s;
  }
  bool is_inverse() const {
    return inverse;   
  }
  virtual size_t size_complex() const = 0;
  virtual size_t size_complex_local() const = 0;
  virtual int compute(::complex* in, ::complex* out) const {}
  virtual int compute(double* in, ::complex* out) const {}
};


class fftw_mpi : public fft_mpi {
protected:
  fftw_plan plan;
public:
  fftw_mpi(const size_t i_n_dimensions, const size_t* i_n, bool i_inverse) : fft_mpi(i_n_dimensions, i_n, i_inverse){
  }
};




class fftw_mpi_c2c : public fftw_mpi{
private:
  fftw_complex *fftw_in, *fftw_out;
public:
  fftw_mpi_c2c(const size_t i_n_dimensions, const size_t* i_n_x, bool i_inverse=false) : fftw_mpi(i_n_dimensions, i_n_x, i_inverse){
    size_t allocated_size;
    if(i_n_dimensions>1){
      allocated_size=2*fftw_mpi_local_size(n_dimensions, n, MPI_COMM_WORLD, &n_x_local, &offset);
      n_x_local_output=n_x_local;
      offset_output=offset;
    }else{
       if(!is_inverse()){
         allocated_size=2*fftw_mpi_local_size_1d(n[0], MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE, &n_x_local, &offset, &n_x_local_output, &offset_output);
       }else{
         allocated_size=2*fftw_mpi_local_size_1d(n[0], MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE, &n_x_local, &offset, &n_x_local_output, &offset_output);  
       }
    }
    fftw_in=fftw_alloc_complex(allocated_size);
    fftw_out=fftw_alloc_complex(allocated_size);
    if(!is_inverse()){
      plan=fftw_mpi_plan_dft(n_dimensions, n, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    }else{
      plan=fftw_mpi_plan_dft(n_dimensions, n, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);  
    }
  }
  ~fftw_mpi_c2c(){
    fftw_destroy_plan(plan);
    // fftw_free(fftw_in);
    // fftw_free(fftw_out);
  }
  size_t size_complex() const {
    return size();
  }
  size_t size_complex_local() const {
    return size_local();
  }
  size_t size_output_local() const {
    if(get_n_dimensions()==1){
      return size_local_output();
    }else{
      if(is_inverse()){
        return size_local();   
      }else{
        return size_complex_local();   
      }
    }
  }
  int compute(::complex* in, ::complex* out){
    size_t i;
    if(!is_inverse()){
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=in[i].real();
        fftw_in[i][1]=in[i].imag();
      }
      //   fftw_execute(plan);
      for(i=offset_output; i<offset_output+size_local(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=out[i].real();
        fftw_in[i][1]=out[i].imag();
      }
      //   fftw_execute(plan);
      for(i=offset_output; i<offset_output+size_local(); ++i){
        in[i].x=fftw_out[i][0]/size();
        in[i].y=fftw_out[i][1]/size();
      }       
    }
    return 0;
  }
};




class fftw_mpi_r2c : public fftw_mpi{
private:
  double* fftw_in;
  fftw_complex* fftw_out;
public:
  fftw_mpi_r2c(const size_t i_n_dimensions, const size_t* i_n_x, bool i_inverse=false) : fftw_mpi(i_n_dimensions, i_n_x, i_inverse){
    size_t allocated_size;
    allocated_size=2*fftw_mpi_local_size(n_dimensions, n, MPI_COMM_WORLD, &n_x_local, &offset);
    fftw_in=(double*)fftw_malloc(sizeof(double)*allocated_size);
    fftw_out=fftw_alloc_complex(allocated_size);
    if(!is_inverse()){
      plan=fftw_mpi_plan_dft_r2c(n_dimensions, n, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_ESTIMATE);
    }else{
      plan=fftw_mpi_plan_dft_c2r(n_dimensions, n, fftw_out, fftw_in, MPI_COMM_WORLD, FFTW_ESTIMATE);  
    }
  }
  ~fftw_mpi_r2c(){
    //    fftw_destroy_plan(plan);
    // fftw_free(fftw_in);
    // fftw_free(fftw_out);
  }
  size_t size_complex() const {
    size_t s=1;
    for(size_t i=0; i<n_dimensions-1; ++i){
      s=s*n[i];
    }
    s=s*(n[n_dimensions-1]/2+1);
    return s;
  }
  size_t size_complex_local() const {
    if(n_dimensions==1){
      return n_x_local/2+1;    
    }
    size_t s=n_x_local;
    for(size_t i=1; i<n_dimensions-1; ++i){
      s=s*n[i];
    }
    s=s*(n[n_dimensions-1]/2+1);
    return s;
  }
  int compute(double* in, ::complex* out){
    size_t i;
    if(!is_inverse()){
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i]=in[i];
      }
      std::cout << "before\n";
      //   fftw_execute(plan);
      std::cout << "after\n";
      for(i=offset; i<offset+size_complex_local(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=offset; i<offset+size_complex_local(); ++i){
        fftw_out[i][0]=out[i].x;
        fftw_out[i][1]=out[i].y;
      }
      //  fftw_execute(plan);
     for(i=offset; i<offset+size_local(); ++i){
       in[i]=fftw_in[i]/size();
     }
    }
    return 0;
  }
};


/*

class ffw_mpi_c2c_1d{
public:
  typedef long size_t;
private:
  fftw_plan plan;
  size_t n_x, n_x_local, n_y, offset, allocated_size;
  bool inverse;
  fftw_complex *fftw_in, *fftw_out;
public:
  inline size_t size_x() const {
    return n_x;
  }
  inline size_t size_x_local() const {
    return n_x_local;
  }
  inline size_t size() const {
    return size_x();
  }
  inline size_t size_local() const {
    return size_x_local();
  }
  inline bool is_inverse() const{
    return inverse;
  }
  ffw_mpi_c2c_1d(const size_t& i_n_x, const bool i_inverse=false){
    n_x=i_n_x;
    inverse=i_inverse;
    size_t sign;
    if(is_inverse()){
      sign=1;
    }else{
      sign=-1;
    }
    allocated_size=fftw_mpi_local_size_1d(n_x, MPI_COMM_WORLD, sign, FFTW_ESTIMATE, &n_x_local, &offset, &n_x_local, &offset);
    fftw_in=fftw_alloc_complex(allocated_size);
    fftw_out=fftw_alloc_complex(allocated_size);
    if(!is_inverse()){
      plan=fftw_mpi_plan_dft_1d(n_x, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    }else{
      plan=fftw_mpi_plan_dft_1d(n_x, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }
  ~ffw_mpi_c2c_1d(){
     fftw_destroy_plan(plan);
     fftw_free(fftw_in);
     fftw_free(fftw_out);
  }
  int compute(::complex* in, ::complex* out){
    size_t i;
    if(!is_inverse()){
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=in[i].real();
        fftw_in[i][1]=in[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=out[i].real();
        fftw_in[i][1]=out[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        in[i].x=fftw_out[i][0]/size();
        in[i].y=fftw_out[i][1]/size();
      }       
    }
    return 0;
  }

};



class ffw_mpi_c2c_2d{
public:
  typedef long size_t;
private:
  fftw_plan plan;
  size_t n_x, n_x_local, n_y, offset, allocated_size;
  bool inverse;
  fftw_complex *fftw_in, *fftw_out;
public:
  inline size_t size_x() const {
    return n_x;
  }
  inline size_t size_x_local() const {
    return n_x_local;
  }
  inline size_t size_y() const {
    return n_y;
  }
  inline size_t size() const {
    return size_x()*size_y();
  }
  inline size_t size_local() const {
    return size_x_local()*size_y();
  }
  inline bool is_inverse() const{
    return inverse;
  }
  ffw_mpi_c2c_2d(const size_t& i_n_x, const size_t& i_n_y, const bool i_inverse=false){
    n_x=i_n_x;
    n_y=i_n_y;
    inverse=i_inverse;
    allocated_size=fftw_mpi_local_size_2d(n_x, n_y, MPI_COMM_WORLD, &n_x_local, &offset);
    fftw_in=fftw_alloc_complex(allocated_size);
    fftw_out=fftw_alloc_complex(allocated_size);
    if(!is_inverse()){
      plan=fftw_mpi_plan_dft_2d(n_x, n_y, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    }else{
      plan=fftw_mpi_plan_dft_2d(n_x, n_y, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }
  ~ffw_mpi_c2c_2d(){
     fftw_destroy_plan(plan);
     fftw_free(fftw_in);
     fftw_free(fftw_out);
  }
  int compute(::complex* in, ::complex* out){
    size_t i;
    if(!is_inverse()){
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=in[i].real();
        fftw_in[i][1]=in[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=out[i].real();
        fftw_in[i][1]=out[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        in[i].x=fftw_out[i][0]/size();
        in[i].y=fftw_out[i][1]/size();
      }       
    }
    return 0;
  }

};

class ffw_mpi_c2c_3d{
public:
  typedef long size_t;
private:
  fftw_plan plan;
  size_t n_x, n_x_local, n_y, n_z, offset, allocated_size;
  bool inverse;
  fftw_complex *fftw_in, *fftw_out;
public:
  inline size_t size_x() const {
    return n_x;
  }
  inline size_t size_x_local() const {
    return n_x_local;
  }
  inline size_t size_y() const {
    return n_y;
  }
  inline size_t size_z() const {
    return n_z;
  }
  inline size_t size() const {
    return size_x()*size_y()*size_z();
  }
  inline size_t size_local() const {
    return size_x_local()*size_y()*size_z();
  }
  inline bool is_inverse() const{
    return inverse;
  }
  ffw_mpi_c2c_3d(const size_t& i_n_x, const size_t& i_n_y, const size_t& i_n_z, const bool i_inverse=false){
    n_x=i_n_x;
    n_y=i_n_y;
    n_z=i_n_z;
    inverse=i_inverse;
    allocated_size=fftw_mpi_local_size_3d(n_x, n_y, n_z, MPI_COMM_WORLD, &n_x_local, &offset);
    fftw_in=fftw_alloc_complex(allocated_size);
    fftw_out=fftw_alloc_complex(allocated_size);
    if(!is_inverse()){
      plan=fftw_mpi_plan_dft_3d(n_x, n_y, n_z, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    }else{
      plan=fftw_mpi_plan_dft_3d(n_x, n_y, n_z, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }
  ~ffw_mpi_c2c_3d(){
     fftw_destroy_plan(plan);
     fftw_free(fftw_in);
     fftw_free(fftw_out);
  }
  int compute(::complex* in, ::complex* out){
    size_t i;
    if(!is_inverse()){
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=in[i].real();
        fftw_in[i][1]=in[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=out[i].real();
        fftw_in[i][1]=out[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        in[i].x=fftw_out[i][0]/size();
        in[i].y=fftw_out[i][1]/size();
      }       
    }
    return 0;
  }

};







    
class ffw_mpi_r2c_1d{
public:
  typedef long size_t;
private:
  fftw_plan plan;
  size_t n_x, n_x_local, n_y, offset, allocated_size;
  bool inverse;
  double *fftw_in;
  fftw_complex *fftw_out;
public:
  inline size_t size_x() const {
    return n_x;
  }
  inline size_t size_x_local() const {
    return n_x_local;
  }
  inline size_t size() const {
    return size_x();
  }
  inline size_t size_local() const {
    return size_x_local();
  }
  inline size_t size_complex() const {
    return size_x()/2+1;
  }
  inline size_t size_complex_local() const {
    return size_x_local()/2+1;
  }
  inline bool is_inverse() const{
    return inverse;
  }
  ffw_mpi_r2c_1d(const size_t& i_n_x, const bool i_inverse=false){
    n_x=i_n_x;
    inverse=i_inverse;
    size_t sign;
    if(is_inverse()){
      sign=1;
    }else{
      sign=-1;
    }
    allocated_size=fftw_mpi_local_size_1d(n_x, MPI_COMM_WORLD, sign, FFTW_ESTIMATE, &n_x_local, &offset, &n_x_local, &offset);

    fftw_in=(double*)fftw_malloc(sizeof(double)*allocated_size);
    fftw_out=fftw_alloc_complex(allocated_size/2+1);
    if(!is_inverse()){
      plan=fftw_mpi_plan_dft_r2c_1d(n_x, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_ESTIMATE);
    }else{
      plan=fftw_mpi_plan_dft_c2r_1d(n_x, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_ESTIMATE);
    }
  }
  ~ffw_mpi_r2c_1d(){
     fftw_destroy_plan(plan);
     fftw_free(fftw_in);
     fftw_free(fftw_out);
  }
  int compute(double* in, ::complex* out) const {
    size_t i;
    if(!is_inverse()){
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i]=in[i];
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_complex_local(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=offset; i<offset+size_complex_local(); ++i){
        fftw_out[i][0]=out[i].x;
        fftw_out[i][1]=out[i].y;
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        in[i]=fftw_in[i]/size();
      }
    }
    return 0;
  }
};



class ffw_mpi_r2c_2d{
public:
  typedef long size_t;
private:
  fftw_plan plan;
  size_t n_x, n_x_local, n_y, offset, allocated_size;
  bool inverse;
  mutable fftw_complex *fftw_in, *fftw_out;
public:
  inline size_t size_x() const {
    return n_x;
  }
  inline size_t size_x_local() const {
    return n_x_local;
  }
  inline size_t size_y() const {
    return n_y;
  }
  inline size_t size() const {
    return size_x()*size_y();
  }
  inline size_t size_local() const {
    return size_x_local()*size_y();
  }
  inline bool is_inverse() const{
    return inverse;
  }
  ffw_mpi_r2c_2d(const size_t& i_n_x, const size_t& i_n_y, const bool i_inverse=false){
    n_x=i_n_x;
    n_y=i_n_y;
    inverse=i_inverse;
    allocated_size=fftw_mpi_local_size_2d(n_x, n_y, MPI_COMM_WORLD, &n_x_local, &offset);
    fftw_in=fftw_alloc_complex(allocated_size);
    fftw_out=fftw_alloc_complex(allocated_size);
    if(!is_inverse()){
      plan=fftw_mpi_plan_dft_2d(n_x, n_y, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    }else{
      plan=fftw_mpi_plan_dft_2d(n_x, n_y, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }
  ~ffw_mpi_r2c_2d(){
     fftw_destroy_plan(plan);
     fftw_free(fftw_in);
     fftw_free(fftw_out);
  }
  int compute(::complex* in, ::complex* out) const {
    size_t i;
    if(!is_inverse()){
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=in[i].real();
        fftw_in[i][1]=in[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=out[i].real();
        fftw_in[i][1]=out[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        in[i].x=fftw_out[i][0]/size();
        in[i].y=fftw_out[i][1]/size();
      }       
    }
    return 0;
  }

};

class ffw_mpi_r2c_3d{
public:
  typedef long size_t;
private:
  fftw_plan plan;
  size_t n_x, n_x_local, n_y, n_z, offset, allocated_size;
  bool inverse;
  mutable fftw_complex *fftw_in, *fftw_out;
public:
  inline size_t size_x() const {
    return n_x;
  }
  inline size_t size_x_local() const {
    return n_x_local;
  }
  inline size_t size_y() const {
    return n_y;
  }
  inline size_t size_z() const {
    return n_z;
  }
  inline size_t size() const {
    return size_x()*size_y()*size_z();
  }
  inline size_t size_local() const {
    return size_x_local()*size_y()*size_z();
  }
  inline bool is_inverse() const{
    return inverse;
  }
  ffw_mpi_r2c_3d(const size_t& i_n_x, const size_t& i_n_y, const size_t& i_n_z, const bool i_inverse=false){
    n_x=i_n_x;
    n_y=i_n_y;
    n_z=i_n_z;
    inverse=i_inverse;
    allocated_size=fftw_mpi_local_size_3d(n_x, n_y, n_z, MPI_COMM_WORLD, &n_x_local, &offset);
    fftw_in=fftw_alloc_complex(allocated_size);
    fftw_out=fftw_alloc_complex(allocated_size);
    if(!is_inverse()){
      plan=fftw_mpi_plan_dft_3d(n_x, n_y, n_z, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    }else{
      plan=fftw_mpi_plan_dft_3d(n_x, n_y, n_z, fftw_in, fftw_out, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }
  ~ffw_mpi_r2c_3d(){
     fftw_destroy_plan(plan);
     fftw_free(fftw_in);
     fftw_free(fftw_out);
  }
  int compute(::complex* in, ::complex* out) const {
    size_t i;
    if(!is_inverse()){
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=in[i].real();
        fftw_in[i][1]=in[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=offset; i<offset+size_local(); ++i){
        fftw_in[i][0]=out[i].real();
        fftw_in[i][1]=out[i].imag();
      }
      fftw_execute(plan);
      for(i=offset; i<offset+size_local(); ++i){
        in[i].x=fftw_out[i][0]/size();
        in[i].y=fftw_out[i][1]/size();
      }       
    }
    return 0;
  }

};

*/

#endif
