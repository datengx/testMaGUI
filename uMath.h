#ifndef __UMATH_H__
#define __UMATH_H__
#include <math.h>

#ifdef WIN32
/* Hacky solution: replace with Boost version of rounding */
double round(double d)
{
  return floor(d + 0.5);
}
float round(float d)
{
  return floor(d + 0.5);
}
#endif

#endif