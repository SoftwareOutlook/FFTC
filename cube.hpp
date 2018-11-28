#ifndef CUBE_HPP
#define CUBE_HPP

#include <omp.h>

template<class T> class cube{
public:
  typedef unsigned long size_t;
private:
  size_t n_x, n_y, n_z;
  T* t;
  inline size_t get_index(const size_t i_x, const size_t i_y, const size_t i_z) const {
    return i_x*n_y*n_z+i_y*n_z+i_z;   
  }
  class proxy{
  private:
    T* p;
    cube::size_t n_z;
  public:
    proxy(T* i_p, cube::size_t i_n_z){
      p=i_p;
      n_z=i_n_z;
    }
    T* operator[](const size_t i_y){
      return p+i_y*n_z;   
    }
  };
public:
  cube(const size_t i_n_x, const size_t i_n_y, const size_t i_n_z){
    n_x=i_n_x;
    n_y=i_n_y;
    n_z=i_n_z;
    t=new T[n_x*n_y*n_z];
  }
  cube(const cube& c) : cube(c.n_x, c.n_y, c.n_z){
    for(size_t i=0; i<n_x*n_y*n_z; ++i){
      t[i]=c.get(i);   
    }
  }
  cube operator=(const cube& c){
    if(this!=&c){
      n_x=c.n_x;
      n_y=c.n_y;
      n_z=c.n_z;
      t=new T[n_x*n_y*n_z];
    }
    return *this;
  }
  cube(cube&& c){
    n_x=c.n_x;
    n_y=c.n_y;
    n_z=c.n_z;
    t=c.t;
    c.t=nullptr;
  } 
  cube operator=(cube&& c){
    if(this!=&c){
      n_x=c.n_x;
      n_y=c.n_y;
      n_z=c.n_z;
      t=c.t;
      c.t=nullptr;
    }
    return *this;
  }
  ~cube(){
    delete[] t;   
  }
  
  inline size_t size_x() const {
    return n_x;   
  }
  inline size_t size_y() const {
    return n_y;   
  }  
  inline size_t size_z() const {
    return n_z;   
  }
  inline size_t size() const {
    return n_x*n_y*n_z;   
  }
  inline proxy operator[](const size_t i_x){
    return proxy(t+i_x*n_y*n_z, n_z);  
  }
  inline T get(const size_t i) const {
    return t[i];   
  }
  inline void set(const size_t i, const T& i_t){
    t[i]=i_t;   
  }
  inline T get(const size_t i_x, const size_t i_y, const size_t i_z) const {
    return t[get_index(i_x, i_y, i_z)];
  }
  inline void set(const size_t i_x, const size_t i_y, const size_t i_z, const T& i_t){
    t[get_index(i_x, i_y, i_z)]=i_t;   
  }
  template<class T2> inline cube operator*(const cube<T2>& c){
    cube p(n_x, n_y, n_z);
    product(*this, c, p);
    return p;
  }
  T* get_pointer_to_slice(const size_t i_x){
    return t+i_x*n_y*n_z;
  }
  T* p(){
    return t;   
  }
};

template<class T1, class T2, class T3> void product(const cube<T1>& a, const cube<T2>& b, cube<T3>& ab){
  for(size_t i=0; i<a.size(); ++i){
    ab.set(i, a.get(i)*b.get(i));
  }
}

#endif
