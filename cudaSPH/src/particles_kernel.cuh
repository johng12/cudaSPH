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

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams
{
    double3 colliderPos;
    double  colliderRadius;

    double3 gravity;
    double globalDamping;
    double particleRadius;

    uint3 gridSize;
    uint numCells;
    double3 worldOrigin;
    double3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    double spring;
    double damping;
    double shear;
    double attraction;
    double boundaryDamping;
};

#endif
