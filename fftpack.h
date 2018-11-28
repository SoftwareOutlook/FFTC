/*
 * fftpack.c : A set of FFT routines in C.
 * Algorithmically based on Fortran-77 FFTPACK by Paul N. Swarztrauber (Version 4, 1985).
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

//#include <Python.h>
#include <math.h>
#include <stdio.h>
//#include <numpy/ndarraytypes.h>

#define DOUBLE
#ifdef DOUBLE
#define Treal double
#else
#define Treal float
#endif

//#define ref(u,a) u[a]

/* Macros for accurate calculation of the twiddle factors. */
#define TWOPI 6.283185307179586476925286766559005768391
#define cos2pi(m, n) cos((TWOPI * (m)) / (n))
#define sin2pi(m, n) sin((TWOPI * (m)) / (n))

#define MAXFAC 13    /* maximum number of factors in factorization of n */
#define NSPECIAL 4   /* number of factors for which we have special-case routines */

#ifdef __cplusplus
extern "C" {
#endif

void sincos2pi(int m, int n, Treal* si, Treal* co);

static void passf2(int ido, int l1, const Treal cc[], Treal ch[], const Treal wa1[], int isign);

static void passf3(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], int isign);

static void passf4(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[], int isign);

static void passf5(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[], const Treal wa4[], int isign);

static void passf(int *nac, int ido, int ip, int l1, int idl1,
      Treal cc[], Treal ch[],
      const Treal wa[], int isign);
  /* isign is -1 for forward transform and +1 for backward transform */
  
void radf2(int ido, int l1, const Treal cc[], Treal ch[], const Treal wa1[]);

void radb2(int ido, int l1, const Treal cc[], Treal ch[], const Treal wa1[]);

void radf3(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[]);

void radb3(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[]);

void radf4(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[]);

void radb4(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[]);

void radf5(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[], const Treal wa4[]);

void radb5(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[], const Treal wa4[]);

void radfg(int ido, int ip, int l1, int idl1,
      Treal cc[], Treal ch[], const Treal wa[]);

void radbg(int ido, int ip, int l1, int idl1,
      Treal cc[], Treal ch[], const Treal wa[]);

void cfftf1(int n, Treal c[], Treal ch[], const Treal wa[], const int ifac[MAXFAC+2], int isign);

void factorize(int n, int ifac[MAXFAC+2], const int ntryh[NSPECIAL]);

void cffti1(int n, Treal wa[], int ifac[MAXFAC+2]);

void rfftf1(int n, Treal c[], Treal ch[], const Treal wa[], const int ifac[MAXFAC+2]);

void rfftb1(int n, Treal c[], Treal ch[], const Treal wa[], const int ifac[MAXFAC+2]);

void rffti1(int n, Treal wa[], int ifac[MAXFAC+2]);

#ifdef __cplusplus
}
#endif
