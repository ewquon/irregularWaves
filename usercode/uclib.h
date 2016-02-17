/*
 * Directly copied from uclib.h in the Star documentation:
 *   Working with User Code 
 *   >> User Function Interface Reference (C)
 *   >> Type Definitions (C)
 */

#ifndef UCLIB_H
#define UCLIB_H
#ifdef DOUBLE_PRECISION
typedef double Real;
#else
typedef float Real;
#endif
typedef double CoordReal;
  
#ifdef __cplusplus
extern "C" {
#endif
#if defined(WIN32) || defined(_WINDOWS) || defined(_WINNT)
# define USERFUNCTION_EXPORT __declspec(dllexport)
# define USERFUNCTION_IMPORT __declspec(dllimport)
#else
# define USERFUNCTION_EXPORT
# define USERFUNCTION_IMPORT
#endif
      
    extern void USERFUNCTION_IMPORT ucarg(void *, char *, char *, int);
    extern void USERFUNCTION_IMPORT ucfunc(void *, char *, char *);
    extern void USERFUNCTION_IMPORT ucfunction(void *, char *, char *, int, ...);
      
    void USERFUNCTION_EXPORT uclib();
#ifdef __cplusplus
}
#endif
#endif

