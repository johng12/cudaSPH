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
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif


typedef unsigned int uint;
enum type_of_particle { FLUID = 3, BOUNDARY = 0};
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
//	std::string working_directory; // directory executable is called from
//	std::string output_directory; // directory where code will place all output files
	Real fixed_dt; // fixed time step value. Default is to use adaptive time stepping
	periodicity periodic_in; // used for periodic boundary conditions. Default is none

};

struct integrate_predictor
{
    Real deltaTime;

    __host__ __device__
    integrate_predictor(Real delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile Real4 posData = thrust::get<0>(t); // position and pressure data
        volatile Real4 velData = thrust::get<1>(t); // velocity and density data
        volatile Real4 posPreData = thrust::get<2>(t); // holds position and pressure data from predictor step
        volatile Real4 velPreData = thrust::get<3>(t); // holds velocity and density data from predictor step
        Real4 accelData = thrust::get<4>(t); // holds dv_dt and drho_dt data

        Real3 pos = make_Real3(posData.x, posData.y, posData.z);
        Real3 vel = make_Real3(velData.x, velData.y, velData.z);
        Real rhop = velData.w;

        Real3 posPre = make_Real3(posPreData.x, posPreData.y, posPreData.z);
		Real3 velPre = make_Real3(velPreData.x, velPreData.y, velPreData.z);
		Real rhopPre = velPreData.w;

		Real3 dvdt = make_Real3(accelData.x,accelData.y,accelData.z);
		Real drhodt = accelData.w;

		// Update density a half time step
		rhopPre = rhop + drhodt * deltaTime * 0.5;

		//Update velocity a half time step
        velPre = vel + dvdt * deltaTime * 0.5;

        // Update position a half time step
        posPre = pos + vel * deltaTime * 0.5;

        // store density, velocity, and position at half step
        thrust::get<2>(t) = make_Real4(posPre,posPreData.w);
        thrust::get<3>(t) = make_Real4(velPre,rhopPre);
    }
};

struct integrate_corrector
{
    Real deltaTime;

    __host__ __device__
    integrate_corrector(Real delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile Real4 posData = thrust::get<0>(t);
        volatile Real4 velData = thrust::get<1>(t);
        volatile Real4 posPreData = thrust::get<2>(t);
        volatile Real4 velPreData = thrust::get<3>(t);
        Real4 accelData = thrust::get<4>(t);

        Real3 pos = make_Real3(posData.x, posData.y, posData.z);
		Real3 vel = make_Real3(velData.x, velData.y, velData.z);
		Real rhop = velData.w;

		Real3 posPre = make_Real3(posPreData.x, posPreData.y, posPreData.z);
		Real3 velPre = make_Real3(velPreData.x, velPreData.y, velPreData.z);
		Real rhopPre = velPreData.w;

		Real3 dvdt = make_Real3(accelData.x,accelData.y,accelData.z);
		Real drhodt = accelData.w;

		// Update density a half step
		rhopPre = rhop + drhodt * deltaTime * 0.5;

		//Update velocity a half time step
        velPre = vel + dvdt * deltaTime * 0.5;

        // Update position a half time step
        posPre = pos + vel * deltaTime * 0.5;

        // Correction Step
        rhop = rhopPre * 2.0 - rhop;
        vel = velPre * 2.0 - vel;
        pos = posPre * 2.0 - pos;

        // store new position and velocity
        thrust::get<0>(t) = make_Real4(pos, posData.w);
        thrust::get<1>(t) = make_Real4(vel, rhop);
    }
};
// C++ Wrappers for CUDA kernels
namespace gpusph
{
	void cudaInit(int argc, char **argv);
	void allocateArray(void **devPtr, size_t size);
	void freeArray(void *devPtr);
	void threadSync();
	void copyArrayToDevice(void *device, const void *host, int offset, int size);
	void copyArrayFromDevice(void *host, const void *device,int offset, int size);
	void set_sim_parameters(simulation_parameters *hostParams);
	void set_domain_parameters(domain_parameters *hostParams);
	void set_exec_parameters(execution_parameters *hostParams);
	uint iDivUp(uint a, uint b);
	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);
	void predictorStep(Real *pospres,
	                         Real *velrhop,
	                         Real *pospres_pre,
	                         Real *velrhop_pre,
	                         Real *ace_drhodt,
	                         Real deltaTime,
	                         uint numParticles);
	void correctorStep(Real *pos,
	                         Real *vel,
	                         Real *pospre,
	                         Real *velpre,
	                         Real *ace_drhodt,
	                         Real deltaTime,
	                         uint numParticles);
	void calcHash(uint  *gridParticleHash,
	                  uint  *gridParticleIndex,
	                  Real *pos,
	                  int    numParticles);
	void reorderDataAndFindCellStart(uint  *cellStart,
	                                     uint  *cellEnd,
	                                     Real *sortedPos,
	                                     Real *sortedVel,
	                                     uint *sortedType,
	                                     uint  *gridParticleHash,
	                                     uint  *gridParticleIndex,
	                                     Real *oldPos,
	                                     Real *oldVel,
	                                     uint *oldType,
	                                     uint   numParticles,
	                                     uint numCells);
	void pre_interaction(Real *ace_drhodt, // output: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
				   Real *velrhop, // input: sorted velocity and density (v.x,v.y,v.z,rhop)
				   Real *pospres, // input: sorted particle positions and pressures
				   Real *viscdt, // output: max time step for adaptive time stepping
				   uint numParticles);
	void compute_interactions(Real *ace_drhodt,
	                 Real *sorted_velrhop,
	                 Real *sorted_pospres,
	                 uint  *gridParticleIndex,
	                 uint  *cellStart,
	                 uint  *cellEnd,
	                 uint   *sorted_type,
	                 Real  *viscdt,
	                 uint   numParticles,
	                 uint   numCells);
	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
	void zero_acceleration(Real *ace_drhodt, uint numParticles);
	void zero_ycomponent(Real *data, uint numParticles);
	Real cuda_max(Real *data, uint numElements);

	__device__ int cellExists(int3 cellPos);
	__device__ int3 calcGridPos(Real3 p);
	__device__ uint calcGridHash(int3 gridPos);
	__global__
	void calcHashD(uint   *gridParticleHash,  // output
	               uint   *gridParticleIndex, // output
	               Real4 *pos,               // input: positions
	               uint    numParticles);
	__global__
	void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
	                                  uint   *cellEnd,          // output: cell end index
	                                  Real4 *sortedPos,        // output: sorted positions
	                                  Real4 *sortedVel,        // output: sorted velocities
	                                  uint  *sortedType,
	                                  uint   *gridParticleHash, // input: sorted grid hashes
	                                  uint   *gridParticleIndex,// input: sorted particle indices
	                                  Real4 *oldPos,           // input: sorted position array
	                                  Real4 *oldVel,           // input: sorted velocity array
	                                  uint  *oldType,
	                                  uint    numParticles);
	__device__
	void particle_particle_interaction(Real4 pospres1, Real4 velrhop1, Real massp1,
									   Real4 pospres2, Real4 velrhop2, Real massp2,
									   Real3 acep1, Real arp1, Real visc);
	__device__
	void interact_with_cell(int3 gridPos, //
							uint index, // index of particle i
							Real  massp1, // mass of particle i
							int   *type, // Ordered particle type data for all particles
							Real4 pospres1, // position vector and pressure of particle i
							Real4 velrhop1, // velocity and density of particle i
							Real4 *pospres, // Ordered position and pressure data for all particles
							Real4 *velrhop, // Ordered velocity and density data for all particles
							Real3 acep1, // Acceleration accumulator for particle i
							Real arp1, // Density derivative accumulator for particle i
							Real  visc, // Max dt for particle i based on viscous considerations
							uint *cellStart, // Index of 1st particle in each grid cell
							uint *cellEnd); // Index of last particle in each grid cell
	__global__
	void compute_particle_interactions(Real4 *ace_drhodt, // output: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
									   Real4 *velrhop, // input: sorted velocity and density (v.x,v.y,v.z,rhop)
									   Real4 *pospres, // input: sorted particle positions and pressures
									   uint *gridParticleIndex, // input: sorted particle indicies
									   uint *cellStart,
									   uint *cellEnd,
									   uint  *type, // input: sorted particle type (e.g. fluid, boundary, etc.)
									   Real *viscdt, // output: max time step for adaptive time stepping
									   uint numParticles);
	__global__
	void pre_interactionD(Real4 *ace_drhodt, // output: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
					   Real4 *velrhop, // input: sorted velocity and density (v.x,v.y,v.z,rhop)
					   Real4 *pospres, // input: sorted particle positions and pressures
					   Real *viscdt, // output: max time step for adaptive time stepping
					   uint numParticles);
	__global__ void zero_accelerationD(Real4 *ace_drhodt);
	__global__ void zero_ycomponentD(Real4 *data, uint numParticles);


}

#endif
