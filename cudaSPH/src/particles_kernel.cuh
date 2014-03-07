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
enum type_of_particle { FLUID, BOUNDARY };
enum simulation_type {ONE_D, TWO_D, THREE_D};
enum periodicity {NONE, IN_X, IN_Y, IN_Z};

// Physical and SPH simulation parameters
struct simulation_parameters
{
	// Numbers of particles
	uint num_particles;
	uint num_fluid_particles;
	uint num_boundary_particles;

	// Parameters used in force computations
	Real3 gravity; // body force per unit mass due to gravity m/s^2
	Real smoothing_length; // m^3
	Real over_smoothing_length; // 1/smoothing_length
	Real four_h_squared; // (2*smoothing_length)^2
	Real rhop0; // Reference density kg/m^3
	Real over_rhop0; // = 1/rho0
	Real fluid_mass; // Mass of a fluid particle, kg
	Real boundary_mass; // Mass of a boundary particle, kg
	Real cs0; // Speed of sound (m/s) at reference density.
	Real wendland_a1,wendland_a2; // Constants for the Wendland kernel
	Real epsilon; // small number used in artificial viscosity model to avoid division by 0
	Real nu; // dynamic viscosity
	Real cfl_number; // courant-friedric-levy number

	// Tait EOS parameters
	Real gamma; // ratio of specific heats
	Real b_coeff; // stiffness parameter used in Tait equation of state
};

// domain parameters
struct domain_parameters
{
	// Uniform Search Grid Parameters
	uint3 grid_size; // Number of grid cells in each dimension (x,y,z)
	uint num_cells; // Total number of grid cells = gridSize.x*gridSize.y*gridSize.z
	Real3 world_origin; // Minimum point of bounding box
	Real3 world_size; // Size of bounding box
	Real3 cell_size; // Dimensions of a single grid cell = 2 * smoothing_length

	// Neighbor Search Statistics
	uint maxParticlesInCell; // Number of particles in most populated grid cell
	uint maxNeighbors; // Largest number of particle interactions
	uint minNeighbors; // Lowest number of particle interactions
	uint aveNeighbors; // Average number of interactions for the system
};

// execution parameters
struct execution_parameters
{
	simulation_type simulation_dimension; // Simulation type (1D, 2D, or 3D)
	Real save_interval; // time between output file writes
	Real print_interval; // time between simulation summary writes to screen
	uint density_renormalization_frequency; // frequency with which to apply sheppard filter, default is 30
	string working_directory; // directory executable is called from
	string output_directory; // directory where code will place all output files
	Real fixed_dt; // fixed time step value. Default is to use adaptive time stepping
	periodicity periodic_in = NONE; // used for periodic bounary conditions. Default is none

};

#endif
