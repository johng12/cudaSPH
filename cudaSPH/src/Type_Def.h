/*
 * Type_Def.h
 *
 *  Created on: Feb 21, 2014
 *      Author: johng12
 */

#ifndef TYPE_DEF_H
#define TYPE_DEF_H

#include "helper_math.h"

#ifndef USE_PRECISION // USE_PRECISION = 0 (SINGLE PRECISION), = 1 (DOUBLE PRECISION)
#define USE_PRECISION 0 // DEFAULTS TO SINGLE PRECISION
#endif

#if(USE_PRECISION) // DOUBLE PRECISION ENABLED

typedef double  Real;
typedef double2	  Real2;
typedef double3  Real3;
typedef double4  Real4;

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Real2 make_Real2(double s){return make_double2(s);}
inline __host__ __device__ Real2 make_Real2(double3 a){return make_double2(a);}
inline __host__ __device__ Real2 make_Real2(int2 a){return make_double2(a);}
inline __host__ __device__ Real2 make_Real2(uint2 a){return make_double2(a);}

inline __host__ __device__ Real3 make_Real3(double s){return make_double3(s);}
inline __host__ __device__ Real3 make_Real3(double2 a){return make_double3(a);}
inline __host__ __device__ Real3 make_Real3(double2 a, double s){return make_double3(a, s);}
inline __host__ __device__ Real3 make_Real3(double4 a){return make_double3(a);}
inline __host__ __device__ Real3 make_Real3(int3 a){return make_double3(a);}
inline __host__ __device__ Real3 make_Real3(uint3 a){return make_double3(a);}

inline __host__ __device__ Real4 make_Real4(double s){return make_double4(s);}
inline __host__ __device__ Real4 make_Real4(double3 a){return make_double4(a);}
inline __host__ __device__ Real4 make_Real4(double3 a, float w){return make_double4(a, w);}
inline __host__ __device__ Real4 make_Real4(int4 a){return make_double4(a);}
inline __host__ __device__ Real4 make_Real4(uint4 a){return make_double4(a);}

#else
typedef float  Real;
typedef float2  Real2;
typedef float3  Real3;
typedef float4  Real4;

inline __host__ __device__ Real2 make_Real2(float s){return make_float2(s);}
inline __host__ __device__ Real2 make_Real2(float3 a){return make_float2(a);}
inline __host__ __device__ Real2 make_Real2(int2 a){return make_float2(a);}
inline __host__ __device__ Real2 make_Real2(uint2 a){return make_float2(a);}

inline __host__ __device__ Real3 make_Real3(float s){return make_float3(s);}
inline __host__ __device__ Real3 make_Real3(float2 a){return make_float3(a);}
inline __host__ __device__ Real3 make_Real3(float2 a, float s){return make_float3(a, s);}
inline __host__ __device__ Real3 make_Real3(float4 a){return make_float3(a);}
inline __host__ __device__ Real3 make_Real3(int3 a){return make_float3(a);}
inline __host__ __device__ Real3 make_Real3(uint3 a){return make_float3(a);}

inline __host__ __device__ Real4 make_Real4(float s){return make_float4(s);}
inline __host__ __device__ Real4 make_Real4(float3 a){return make_float4(a);}
inline __host__ __device__ Real4 make_Real4(float3 a, float w){return make_float4(a, w);}
inline __host__ __device__ Real4 make_Real4(int4 a){return make_float4(a);}
inline __host__ __device__ Real4 make_Real4(uint4 a){return make_float4(a);}

#endif

#endif
