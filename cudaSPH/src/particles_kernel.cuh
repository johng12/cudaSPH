/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#define USE_TEX 0

#include "vector_types.h"
#include "Type_Def.h"

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif


typedef unsigned int uint;

// simulation parameters
struct SimParams
{
    Real3 colliderPos;
    Real  colliderRadius;
    Real smoothingLength;

    Real3 gravity;
    Real globalDamping;
    Real particleRadius;

    uint3 gridSize;
    uint numCells;
    Real3 worldOrigin;
    Real3 worldSize;
    Real3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    Real spring;
    Real damping;
    Real shear;
    Real attraction;
    Real boundaryDamping;
};

__device__ int cellExists(int3 cellPos);
#endif
