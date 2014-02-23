/*
 * readDataType.h
 *
 *  Created on: Feb 21, 2014
 *      Author: johng12
 */

#ifndef REALDATATYPE_H
#define REALDATATYPE_H

#define PRECISION 0
#if(PRECISION)

typedef double  Real;
typedef double2 Real2;
typedef double3 Real3;
typedef double4 Real4;

#define make_Real2(x,y) make_double2(x,y)
#define make_Real3(x,y,z) make_double3(x,y,z)
#define make_Real4(x,y,z,w) make_double4(x,y,z,w)

#else
typedef float  Real;
typedef float2 Real2;
typedef float3 Real3;
typedef float4 Real4;

#define make_Real2(x,y) make_float2(x,y)
#define make_Real3(x,y,z) make_float3(x,y,z)
#define make_Real4(x,y,z,w) make_float4(x,y,z,w)
#endif

#endif
