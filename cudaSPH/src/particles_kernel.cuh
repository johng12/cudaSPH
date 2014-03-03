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
	// Uniform Search Grid Parameters
	uint3 gridSize; // Number of grid cells in each dimension (x,y,z)
	uint numCells; // Total number of grid cells = gridSize.x*gridSize.y*gridSize.z
	Real3 worldOrigin; // Minimum point of bounding box
	Real3 worldSize; // Size of bounding box
	Real3 cellSize; // Dimensions of a single grid cell = 2 * smoothingLength

	// Neighbor Search Statistics
	uint maxParticlesInCell; // Number of particles in most populated grid cell
	uint maxNeighbors; // Largest number of particle interactions
	uint minNeighbors; // Lowest number of particle interactions
	uint aveNeighbors; // Average number of interactions for the system

	// Types of Particles
	int FLUID;
	int BOUNDARY;

	// Parameters used in force computations
	Real3 gravity;
	Real smoothingLength; // m^3
	Real overSmoothingLength; // 1/smoothingLength
	Real rho0; // Reference density kg/m^3
	Real overRho0; // = 1/rho0
	Real massFluid; // Mass of a fluid particle, kg
	Real massBoundary; // Mass of a boundary particle, kg
	Real cs0; // Speed of sound (m/s) at reference density.
	Real wendland_a1,wendland_a2; // Constants for the Wendland kernel
	Real four_h_squared;
	Real eta2;
	Real visco;

	// Stuff associated with NVIDIA sample - to be deleted in future.
    Real3 colliderPos;
    Real  colliderRadius;

    Real globalDamping;
    Real particleRadius;

    uint numBodies;
    uint maxParticlesPerCell;

    Real spring;
    Real damping;
    Real shear;
    Real attraction;
    Real boundaryDamping;
};

#endif
