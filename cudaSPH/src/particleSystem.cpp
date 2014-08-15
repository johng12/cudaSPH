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

#include "particleSystem.h"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654
#endif


ParticleSystem::ParticleSystem(const char *cfgFileName):

    initialized_(false),
    numParticles_(0),

    h_pospres_(0),
    h_velrhop_(0),
    h_particle_type_(0),
    h_particle_hash_(0),
    h_particle_index_(0),
    h_particle_gridPos_(0),
    h_cell_start_(0),
    h_cell_end_(0),

    d_pospres_(0),
    d_velrhop_(0),
    d_pospres_pre_(0),
	d_velrhop_pre_(0),
	d_ace_drho_dt_(0),
	d_particle_type_(0),
	d_sorted_pospres_(0),
	d_sorted_velrhop_(0),
	d_sorted_type_(0),
	d_velxcor_(0),
	d_visc_dt_(0),
	d_force_dt_(0),
	d_max_accel_(0),
	d_max_sound_speed_(0),
	d_norm_ace_(0),

	d_density_sum_(0),
	d_kernel_sum_(0),

	d_particle_hash_(0),
	d_particle_index_(0),
	d_particle_gridPos_(0),
	d_cell_start_(0),
	d_cell_end_(0),

    m_timer(NULL)
{
	loadCfg(cfgFileName);
    _initialize();


}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    numParticles_ = 0;
}

void
ParticleSystem::_initialize()
{
    assert(!initialized_);

    numParticles_ = h_simulation_params_.num_particles;

    // allocate host storage
    h_pospres_ = new Real[numParticles_*4];
    h_velrhop_ = new Real[numParticles_*4];
    memset(h_pospres_, 0, numParticles_*4*sizeof(Real));
    memset(h_velrhop_, 0, numParticles_*4*sizeof(Real));

    h_particle_type_ = new uint[numParticles_];
		memset(h_particle_type_, 0, numParticles_*sizeof(uint));

	h_particle_hash_ = new uint[numParticles_];
		memset(h_particle_hash_, 0, numParticles_*sizeof(uint));

	h_particle_gridPos_ = new int[numParticles_*4];
		memset(h_particle_gridPos_,0,numParticles_*4*sizeof(int));

	h_particle_index_ = new uint[numParticles_];
		memset(h_particle_index_, 0, numParticles_*sizeof(uint));

    h_cell_start_ = new uint[h_domain_params_.num_cells];
    	memset(h_cell_start_, 0, h_domain_params_.num_cells*sizeof(uint));

    h_cell_end_ = new uint[h_domain_params_.num_cells];
    	memset(h_cell_end_, 0, h_domain_params_.num_cells*sizeof(uint));

	h_neighbors_ = new uint[numParticles_];
		memset(h_neighbors_,0,numParticles_*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(Real) * 4 * numParticles_;

	gpusph::allocateArray((void **)&d_pospres_, memSize);
	gpusph::allocateArray((void **)&d_velrhop_, memSize);
	gpusph::allocateArray((void **)&d_pospres_pre_, memSize);
	gpusph::allocateArray((void **)&d_velrhop_pre_, memSize);
	gpusph::allocateArray((void **)&d_ace_drho_dt_, memSize);
	gpusph::allocateArray((void **)&d_velxcor_,memSize);
	gpusph::allocateArray((void **)&d_particle_type_, sizeof(uint) * numParticles_);
	gpusph::allocateArray((void **)&d_sorted_pospres_, memSize);
	gpusph::allocateArray((void **)&d_sorted_velrhop_, memSize);
	gpusph::allocateArray((void **)&d_sorted_type_, sizeof(uint) * numParticles_);

	gpusph::allocateArray((void **)&d_visc_dt_,sizeof(Real) * numParticles_);
	gpusph::allocateArray((void **)&d_force_dt_,sizeof(Real)* numParticles_ );
	gpusph::allocateArray((void **)&d_norm_ace_,sizeof(Real) * numParticles_);
	gpusph::allocateArray((void **)&d_max_accel_,sizeof(Real));
	gpusph::allocateArray((void **)&d_max_sound_speed_,sizeof(Real));

	gpusph::allocateArray((void **)&d_density_sum_,sizeof(Real) * numParticles_);
	gpusph::allocateArray((void **)&d_kernel_sum_, sizeof(Real) * numParticles_);

	gpusph::allocateArray((void **)&d_particle_hash_, numParticles_*sizeof(uint));
	gpusph::allocateArray((void **)&d_particle_index_, numParticles_*sizeof(uint));
	gpusph::allocateArray((void **)&d_particle_gridPos_,numParticles_*4*sizeof(int));
	gpusph::allocateArray((void **)&d_cell_start_, h_domain_params_.num_cells*sizeof(uint));
	gpusph::allocateArray((void **)&d_cell_end_, h_domain_params_.num_cells*sizeof(uint));
	gpusph::allocateArray((void **)&d_neighbors_,sizeof(uint) * numParticles_);
    sdkCreateTimer(&m_timer);


    initialized_ = true;
}

void
ParticleSystem::_finalize()
{
    assert(initialized_);

    delete [] h_pospres_;
    delete [] h_velrhop_;
    delete [] h_particle_type_;
    delete [] h_particle_hash_;
    delete [] h_particle_gridPos_;
    delete [] h_particle_index_;
    delete [] h_cell_start_;
    delete [] h_cell_end_;

    gpusph::freeArray(d_pospres_);
    gpusph::freeArray(d_velrhop_);
    gpusph::freeArray(d_pospres_pre_);
    gpusph::freeArray(d_velrhop_pre_);
    gpusph::freeArray(d_ace_drho_dt_);
    gpusph::freeArray(d_particle_type_);
    gpusph::freeArray(d_sorted_pospres_);
    gpusph::freeArray(d_sorted_velrhop_);
    gpusph::freeArray(d_sorted_type_);
    gpusph::freeArray(d_velxcor_);

    gpusph::freeArray(d_visc_dt_);
    gpusph::freeArray(d_force_dt_);
    gpusph::freeArray(d_max_accel_);
    gpusph::freeArray(d_max_sound_speed_);

    gpusph::freeArray(d_density_sum_);
    gpusph::freeArray(d_kernel_sum_);

    gpusph::freeArray(d_particle_hash_);
    gpusph::freeArray(d_particle_index_);
    gpusph::freeArray(d_particle_gridPos_);
    gpusph::freeArray(d_cell_start_);
    gpusph::freeArray(d_cell_end_);
    gpusph::freeArray(d_neighbors_);
}

// step the simulation and return the current time step
Real
ParticleSystem::update(Real &deltaTime)
{
    assert(initialized_);
//    deltaTime = h_simulation_params_.cfl_number * h_simulation_params_.smoothing_length/h_simulation_params_.cs0;
    deltaTime = 3.0e-4;
    h_exec_params_.fixed_dt = deltaTime;

    // update constants
    gpusph::set_domain_parameters(&h_domain_params_);
	gpusph::set_sim_parameters(&h_simulation_params_);
    gpusph::set_exec_parameters(&h_exec_params_);

    {//======== Predictor Step =============
		// calculate grid hash
		gpusph::calcHash(
			d_particle_hash_,
			d_particle_index_,
			d_pospres_,
			h_simulation_params_.num_particles,
			d_particle_gridPos_);

//		gpusph::copyArrayFromDevice(h_particle_hash_,d_particle_hash_,0,sizeof(uint)*numParticles_);
//		gpusph::copyArrayFromDevice(h_particle_gridPos_,d_particle_gridPos_,0,sizeof(int)*4*numParticles_);
//		FILE *pFile;
//		pFile = fopen ("gpu_gridPositions.asc","w");
//		for (uint i=0; i<numParticles_; i++)
//		{
//			fprintf(pFile,"%d %d %d\n", h_particle_gridPos_[i*4+0],h_particle_gridPos_[i*4+1],h_particle_gridPos_[i*4+2]);
//		}
//		fclose(pFile);
//
//		pFile = fopen ("gpu_gridHash.asc","w");
//		for (uint i=0; i<numParticles_; i++)
//		{
//			fprintf(pFile,"%d \n", h_particle_hash_[i]);
//		}
//		fclose(pFile);

		// sort particles based on hash
		gpusph::sortParticles(d_particle_hash_, d_particle_index_, h_simulation_params_.num_particles);

//		gpusph::copyArrayFromDevice(h_particle_hash_, d_particle_hash_, 0, sizeof(uint)*numParticles_);
//		gpusph::copyArrayFromDevice(h_particle_index_, d_particle_index_, 0, sizeof(uint)*numParticles_);
//		pFile = fopen ("gpu_sortedGridHash.asc","w");
//		for (uint i=0; i<numParticles_; i++)
//		{
//			fprintf(pFile,"%d %d \n", h_particle_hash_[i],h_particle_index_[i]);
//		}
//		fclose(pFile);

		// reorder particle arrays into sorted order and
		// find start and end of each cell
		gpusph::reorderDataAndFindCellStart(
			d_cell_start_,
			d_cell_end_,
			d_sorted_pospres_,
			d_sorted_velrhop_,
			d_sorted_type_,
			d_particle_hash_,
			d_particle_index_,
			d_pospres_,
			d_velrhop_,
			d_particle_type_,
			h_simulation_params_.num_particles,
			h_domain_params_.num_cells);

//		gpusph::copyArrayFromDevice(h_cell_start_, d_cell_start_, 0, sizeof(uint)*h_domain_params_.num_cells);
//		gpusph::copyArrayFromDevice(h_cell_end_,d_cell_end_,0,sizeof(uint)*h_domain_params_.num_cells);
//
//		pFile = fopen ("gpu_cellData.asc","w");
//		for (uint i=0; i<h_domain_params_.num_cells; i++)
//		{
//			fprintf(pFile,"%d %d\n", h_cell_start_[i],h_cell_end_[i]);
//		}
//		fclose(pFile);
//
//		gpusph::copyArrayFromDevice(h_pospres_, d_sorted_pospres_, 0, sizeof(Real)*4*numParticles_);
//		gpusph::copyArrayFromDevice(h_velrhop_, d_sorted_velrhop_, 0, sizeof(Real)*4*numParticles_);
//		gpusph::copyArrayFromDevice(h_particle_index_, d_particle_index_, 0, sizeof(uint)*numParticles_);
//
//		pFile = fopen ("gpu_sortedData.asc","w");
//
//		Real x,y,z,u,v,w,rho_temp,p_temp;
//		uint index_temp;
//
//		for( uint i = 0; i < h_simulation_params_.num_particles; i++)
//			{
//				x = h_pospres_[4*i];
//				y = h_pospres_[4*i+1];
//				z = h_pospres_[4*i+2];
//				p_temp = h_pospres_[4*i+3];
//
//				u = h_velrhop_[4*i];
//				v = h_velrhop_[4*i+1];
//				w = h_velrhop_[4*i+2];
//				rho_temp = h_velrhop_[4*i+3];
//
//				index_temp = h_particle_index_[i];
//				fprintf(pFile,"%17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %d \n", x,y,z,u,v,w,rho_temp,p_temp,index_temp);
//
//
//			}
//		fclose(pFile);

		// prepare variables for interaction
		// zero accel arrays, get pressures, etc.
		// prepare data for interactions
		gpusph::pre_interaction(d_ace_drho_dt_,
						d_sorted_velrhop_,
						d_sorted_pospres_,
						d_visc_dt_,
						h_simulation_params_.num_particles);


		// process particle interactions
		gpusph::compute_interactions(d_ace_drho_dt_,
							 d_velxcor_,
							 d_sorted_velrhop_,
							 d_sorted_pospres_,
							 d_particle_index_,
							 d_cell_start_,
							 d_cell_end_,
							 d_sorted_type_,
							 d_visc_dt_,
							 h_simulation_params_.num_particles,
							 h_domain_params_.num_cells,
							 d_neighbors_,
							 d_particle_hash_);

//		gpusph::copyArrayFromDevice(h_particle_type_, d_neighbors_, 0, sizeof(uint)*numParticles_);
//		pFile = fopen ("gpu_neighbors.asc","w");
//		for( uint i = 0; i < h_simulation_params_.num_particles; i++)
//		{
//			fprintf(pFile,"%d \n", h_particle_type_[i]);
//		}
//		fclose(pFile);


//		// get time step
//		deltaTime = get_time_step(d_visc_dt_,numParticles_);

		// zero out acceleration of stationary particles.
//		gpusph::zero_acceleration(d_ace_drho_dt_,numParticles_);

		// zero out y-component of data for 2D simulations
//		if(h_exec_params_.simulation_dimension == TWO_D)
//			{
//				gpusph::zero_ycomponent(d_ace_drho_dt_,numParticles_);
//				gpusph::zero_ycomponent(d_velxcor_,numParticles_);
//			}

//		gpusph::copyArrayFromDevice(h_pospres_, d_ace_drho_dt_, 0, sizeof(Real)*4*numParticles_);
//
//		pFile = fopen ("gpu_aceData.asc","w");
//		for (uint i=0; i<numParticles_; i++)
//		{
//			fprintf(pFile,"%17.16e %17.16e %17.16e %17.16e\n", h_pospres_[i*4+0],h_pospres_[i*4+1],h_pospres_[i*4+2],h_pospres_[i*4+3]);
//		}
//		fclose(pFile);

//		gpusph::reduceAccel(d_ace_drho_dt_,d_velrhop_, d_force_dt_,d_visc_dt_,numParticles_);
//		deltaTime = gpusph::get_time_step(d_force_dt_,d_visc_dt_,numParticles_);
//
//		gpusph::set_exec_parameters(&h_exec_params_);

		// predictor step
		gpusph::predictorStep(
			d_pospres_,
			d_velrhop_,
			d_pospres_pre_,
			d_velrhop_pre_,
			d_velxcor_,
			d_ace_drho_dt_,
			h_simulation_params_.num_particles);

//		gpusph::copyArrayFromDevice(h_pospres_, d_pospres_pre_, 0, sizeof(Real)*4*numParticles_);
//		gpusph::copyArrayFromDevice(h_velrhop_, d_velrhop_pre_, 0, sizeof(Real)*4*numParticles_);
//
//		pFile = fopen ("gpu_predictorStep.asc","w");
//		for( uint i = 0; i < h_simulation_params_.num_particles; i++)
//				{
//					x = h_pospres_[4*i];
//					y = h_pospres_[4*i+1];
//					z = h_pospres_[4*i+2];
//
//					u = h_velrhop_[4*i];
//					v = h_velrhop_[4*i+1];
//					w = h_velrhop_[4*i+2];
//					rho_temp = h_velrhop_[4*i+3];
//
//					fprintf(pFile,"%17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e\n", x,y,z,u,v,w,rho_temp);
//
//
//				}
//		fclose(pFile);

    }

    {//============ Corrector Step =============

		FILE *pFile;
		// calculate grid hash
		gpusph::calcHash(
			d_particle_hash_,
			d_particle_index_,
			d_pospres_pre_,
			h_simulation_params_.num_particles,
			d_particle_gridPos_);

		// sort particles based on hash
		gpusph::sortParticles(d_particle_hash_, d_particle_index_, h_simulation_params_.num_particles);

		// reorder particle arrays into sorted order and
		// find start and end of each cell
		gpusph::reorderDataAndFindCellStart(
			d_cell_start_,
			d_cell_end_,
			d_sorted_pospres_,
			d_sorted_velrhop_,
			d_sorted_type_,
			d_particle_hash_,
			d_particle_index_,
			d_pospres_pre_,
			d_velrhop_pre_,
			d_particle_type_,
			h_simulation_params_.num_particles,
			h_domain_params_.num_cells);


		// prepare data for interactions
		gpusph::pre_interaction(d_ace_drho_dt_,
						d_sorted_velrhop_,
						d_sorted_pospres_,
						d_visc_dt_,
						numParticles_);

		// process particle interactions
		gpusph::compute_interactions(d_ace_drho_dt_,
							 d_velxcor_,
    						 d_sorted_velrhop_,
							 d_sorted_pospres_,
							 d_particle_index_,
							 d_cell_start_,
							 d_cell_end_,
							 d_sorted_type_,
							 d_visc_dt_,
							 numParticles_,
							 h_domain_params_.num_cells,
							 d_neighbors_,
							 d_particle_hash_);


//		gpusph::copyArrayFromDevice(h_particle_type_, d_neighbors_, 0, sizeof(uint)*numParticles_);
//		pFile = fopen ("gpu_correctorNeighbors.asc","w");
//		for( uint i = 0; i < h_simulation_params_.num_particles; i++)
//		{
//			fprintf(pFile,"%d \n", h_particle_type_[i]);
//		}
//		fclose(pFile);
//
//		gpusph::copyArrayFromDevice(h_pospres_, d_sorted_pospres_, 0, sizeof(Real)*4*numParticles_);
//		pFile = fopen ("gpu_correctorPospresData.asc","w");
//		for (uint i=0; i<numParticles_; i++)
//		{
//			fprintf(pFile,"%17.16e %17.16e %17.16e %17.16e\n", h_pospres_[i*4+0],h_pospres_[i*4+1],h_pospres_[i*4+2],h_pospres_[i*4+3]);
//		}
//		fclose(pFile);
//
//		gpusph::copyArrayFromDevice(h_pospres_, d_sorted_velrhop_, 0, sizeof(Real)*4*numParticles_);
//		pFile = fopen ("gpu_correctorVelrhopData.asc","w");
//		for (uint i=0; i<numParticles_; i++)
//		{
//			fprintf(pFile,"%17.16e %17.16e %17.16e %17.16e\n", h_pospres_[i*4+0],h_pospres_[i*4+1],h_pospres_[i*4+2],h_pospres_[i*4+3]);
//		}
//		fclose(pFile);

		// zero out acceleration of stationary particles.
		gpusph::zero_acceleration(d_ace_drho_dt_,numParticles_);

		// zero out y-component of data for 2D simulations. still need to add C wrapper to this as well
		if(h_exec_params_.simulation_dimension == TWO_D) {
			gpusph::zero_ycomponent(d_ace_drho_dt_,numParticles_);
			gpusph::zero_ycomponent(d_velxcor_,numParticles_);
		}


//		gpusph::copyArrayFromDevice(h_pospres_, d_ace_drho_dt_, 0, sizeof(Real)*4*numParticles_);
//
//		pFile = fopen ("gpu_correctorAceData.asc","w");
//		for (uint i=0; i<numParticles_; i++)
//		{
//			fprintf(pFile,"%17.16e %17.16e %17.16e %17.16e\n", h_pospres_[i*4+0],h_pospres_[i*4+1],h_pospres_[i*4+2],h_pospres_[i*4+3]);
//		}
//		fclose(pFile);

		// corrector step
        gpusph::correctorStep(
            d_pospres_,
            d_velrhop_,
            d_pospres_pre_,
            d_velrhop_pre_,
            d_ace_drho_dt_,
            numParticles_);

//		gpusph::copyArrayFromDevice(h_pospres_, d_pospres_, 0, sizeof(Real)*4*numParticles_);
//		gpusph::copyArrayFromDevice(h_velrhop_, d_velrhop_, 0, sizeof(Real)*4*numParticles_);
//
//
//		Real x,y,z,u,v,w,rho_temp;
//		pFile = fopen ("gpu_correctorStep.asc","w");
//		for( uint i = 0; i < h_simulation_params_.num_particles; i++)
//				{
//					x = h_pospres_[4*i];
//					y = h_pospres_[4*i+1];
//					z = h_pospres_[4*i+2];
//
//					u = h_velrhop_[4*i];
//					v = h_velrhop_[4*i+1];
//					w = h_velrhop_[4*i+2];
//					rho_temp = h_velrhop_[4*i+3];
//
//					fprintf(pFile,"%17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e \n", x,y,z,u,v,w,rho_temp);
//
//
//				}
//		fclose(pFile);
    }

    return deltaTime;


}

void
ParticleSystem::apply_sheppard_filter()
{
	assert(initialized_);

	// calculate grid hash
	gpusph::calcHash(
		d_particle_hash_,
		d_particle_index_,
		d_pospres_,
		numParticles_,
		d_particle_gridPos_);

	// sort particles based on hash
	gpusph::sortParticles(d_particle_hash_, d_particle_index_, numParticles_);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	gpusph::reorderDataAndFindCellStart(
		d_cell_start_,
		d_cell_end_,
		d_sorted_pospres_,
		d_sorted_velrhop_,
		d_sorted_type_,
		d_particle_hash_,
		d_particle_index_,
		d_pospres_,
		d_velrhop_,
		d_particle_type_,
		numParticles_,
		h_domain_params_.num_cells);

	// prepare variables for interaction
	// zero accel arrays, get pressures, etc.
	// prepare data for interactions
	gpusph::pre_interaction(d_ace_drho_dt_,
					d_sorted_velrhop_,
					d_sorted_pospres_,
					d_visc_dt_,
					numParticles_);


	// process particle interactions
	gpusph::sheppard_density_filter(d_velrhop_,
									d_pospres_,
									d_sorted_velrhop_,
									d_sorted_pospres_,
									d_particle_index_,
									d_cell_start_,
									d_cell_end_,
									d_sorted_type_,
									numParticles_);

}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
	gpusph::copyArrayFromDevice(h_cell_start_, d_cell_start_, 0, sizeof(uint)*h_domain_params_.num_cells);
	gpusph::copyArrayFromDevice(h_cell_end_, d_cell_end_, 0, sizeof(uint)*h_domain_params_.num_cells);
    uint maxCellSize = 0;

    for (uint i=0; i<h_domain_params_.num_cells; i++)
    {
        if (h_cell_start_[i] != 0xffffffff)
        {
            uint cellSize = h_cell_end_[i] - h_cell_start_[i];

            //            printf("cell: %d, %d particles\n", i, cell_size);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParameters()
{
    // Print Simulation Details Summary
	printf("\n\nComputational Domain Details: \n");
	printf("World Origin = %f %f %f \n",h_domain_params_.world_origin.x,h_domain_params_.world_origin.y,h_domain_params_.world_origin.z);
	printf("World Limits = %f %f %f \n",h_domain_params_.world_limits.x,h_domain_params_.world_limits.y,h_domain_params_.world_limits.z);
	printf("Grid Size = %d %d %d \n",h_domain_params_.grid_size.x,h_domain_params_.grid_size.y,h_domain_params_.grid_size.z);
	printf("Cell Size = %5.4f x %5.4f x %5.4f \n",h_domain_params_.cell_size.x, h_domain_params_.cell_size.y,h_domain_params_.cell_size.z);
	printf("Total Cells = %d \n",h_domain_params_.num_cells);
	printf("\n");
	printf("Simulation Details: \n");
	printf("Total Particles = %d, Total Fluid Particles = %d, Total Boundary Particles = %d \n",h_simulation_params_.num_particles,h_simulation_params_.num_fluid_particles,h_simulation_params_.num_boundary_particles);
	printf("Gravity = (%f %f %f) \n",h_simulation_params_.gravity.x,h_simulation_params_.gravity.y,h_simulation_params_.gravity.z);
	printf("dp = %f, h = %f \n",h_simulation_params_.dp,h_simulation_params_.smoothing_length);
	printf("rhop0 = %f, Cs0 = %f, CFL Number = %f \n",h_simulation_params_.rhop0,h_simulation_params_.cs0,h_simulation_params_.cfl_number);
	printf("Fluid particle mass: %f, Boundary particle mass: %f \n",h_simulation_params_.fluid_mass,h_simulation_params_.boundary_mass);
	printf("Artificial viscosity: nu = %f, epsilon = %f  \n",h_simulation_params_.nu,h_simulation_params_.epsilon);
	printf("XSPH coefficient: eps = %f \n",h_simulation_params_.eps);
	printf("Tait EOS: B = %f, gamma = %f  \n",h_simulation_params_.b_coeff,h_simulation_params_.gamma);
	printf("Wendland Coefficients: a1 = %f, a2 = %f \n",h_simulation_params_.wendland_a1,h_simulation_params_.wendland_a2);
	printf("\n");
	printf("Execution Details: \n");
	printf("Simulation Dimension (1D, 2D or 3D): %dD \n",h_exec_params_.simulation_dimension+1);
	printf("Sheppard Steps = %d \n",h_exec_params_.density_renormalization_frequency);
	printf("Simulation Duration = %f s \n",h_exec_params_.simulation_duration);
	printf("Save Interval = %f s \n",h_exec_params_.save_interval);
	printf("\n\n");

}

void
ParticleSystem::dumpParticles(uint start, uint count, Real current_time, const char *fileName)
{
	  FILE * pFile;
	  pFile = fopen (fileName,"w");
	  fprintf(pFile,"VARIABLES=X(m),Y(m),Z(m),U(m/s),V(m/s),W(m/s),Rho(kg/m3),P(Pa) \n");
	  fprintf(pFile,"Zone T=\"BoundaryParticles\",DATAPACKING = POINT,SolutionTime = %10.9e, StrandID = 1 \n",current_time);

    // debug
	  gpusph::copyArrayFromDevice(h_pospres_, d_pospres_, 0, sizeof(Real)*4*count);
	  gpusph::copyArrayFromDevice(h_velrhop_, d_velrhop_, 0, sizeof(Real)*4*count);

    for (uint i=0; i<h_simulation_params_.num_boundary_particles; i++)  {
    	fprintf(pFile,"%10.9e, %10.9e, %10.9e %10.9e, %10.9e, %10.9e %10.9e %10.9e \n", h_pospres_[i*4+0], h_pospres_[i*4+1], h_pospres_[i*4+2], h_velrhop_[i*4+0], h_velrhop_[i*4+1], h_velrhop_[i*4+2], h_velrhop_[i*4+3], h_pospres_[i*4+3]);

    }

	  fprintf(pFile,"VARIABLES=X(m),Y(m),Z(m),U(m/s),V(m/s),W(m/s),Rho(kg/m3),P(Pa) \n");
	  fprintf(pFile,"Zone T=\"FluidParticles\",DATAPACKING = POINT,SolutionTime = %10.9e, StrandID = 2 \n",current_time);


  for (uint i=h_simulation_params_.num_boundary_particles; i<numParticles_; i++)  {
	  fprintf(pFile,"%10.9e, %10.9e, %10.9e %10.9e, %10.9e, %10.9e %10.9e %10.9e\n", h_pospres_[i*4+0], h_pospres_[i*4+1], h_pospres_[i*4+2], h_velrhop_[i*4+0], h_velrhop_[i*4+1], h_velrhop_[i*4+2], h_velrhop_[i*4+3], h_pospres_[i*4+3]);
  }
    fclose(pFile);
}

//Real *
//ParticleSystem::getArray(ParticleArray array)
//{
//    assert(initialized_);
//
//
//	Real *hdata = 0;
//	Real *ddata = 0;
//
//    switch (array)
//    {
//        default:
//        case POSITION:
//            hdata = h_pospres_;
//            ddata = d_pospres_;
//            copyArrayFromDevice(hdata, ddata, 0,numParticles_*4*sizeof(Real));
//            break;
//
//        case VELOCITY:
//            hdata = h_velrhop_;
//            ddata = d_velrhop_;
//            copyArrayFromDevice(hdata, ddata, 0,numParticles_*4*sizeof(Real));
//            break;
//
//    }
//
//    return hdata;
//
//}


//void
//ParticleSystem::setArray(ParticleArray array, const Real *data, int start, int count)
//{
//    assert(initialized_);
//
//    switch (array)
//    {
//        default:
//        case POSITION:
//			copyArrayToDevice(d_pospres_, data, start*4*sizeof(Real), count*4*sizeof(Real));
//			break;
//
//
//
//        case VELOCITY:
//            copyArrayToDevice(d_velrhop_, data, start*4*sizeof(Real), count*4*sizeof(Real));
//            break;
//    }
//}

inline Real frand()
{
    return rand() / (Real) RAND_MAX;
}

void
ParticleSystem::loadCfg(const char *fileName)
{

	TiXmlDocument doc(fileName);
	if (!doc.LoadFile()) return;

	TiXmlHandle hDoc(&doc);
	TiXmlElement *pElem;
	TiXmlHandle hRoot(0);
	std::string m_name;

	// block: name
	{
		pElem = hDoc.FirstChildElement().Element();
		// should always have a valid root but handle gracefully if it does not
		if (!pElem) return;
		m_name = pElem->Value();

		// save this for later
		hRoot = TiXmlHandle(pElem);
	}

#if(USE_PRECISION)
	// block: definition
	{

		pElem = hRoot.FirstChild( "casedef" ).FirstChild( "geometry" ).FirstChild( "definition" ).Element();
		pElem = pElem -> FirstChildElement();
		for (pElem; pElem; pElem = pElem -> NextSiblingElement())
		{
			if (!strcmp("pointmin",pElem->Value()))
			{
				pElem ->QueryDoubleAttribute("x", &h_domain_params_.world_origin.x);
				pElem ->QueryDoubleAttribute("y", &h_domain_params_.world_origin.y);
				pElem ->QueryDoubleAttribute("z", &h_domain_params_.world_origin.z);
			}
			else if (!strcmp("pointmax",pElem->Value()))
			{
				pElem ->QueryDoubleAttribute("x", &h_domain_params_.world_limits.x);
				pElem ->QueryDoubleAttribute("y", &h_domain_params_.world_limits.y);
				pElem ->QueryDoubleAttribute("z", &h_domain_params_.world_limits.z);
			}

		}
	}

	// block: execution
	{
		TiXmlElement *child;
		const char *key;
		double value;
		pElem = hRoot.FirstChild( "execution").Element();

		// parameters
		child = pElem -> FirstChildElement("parameters")->FirstChildElement();
		for(child;child;child = child -> NextSiblingElement())
		{
			key = child ->Attribute("key");
			if(!strcmp(key,"Visco"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.nu);
			}

			if(!strcmp(key,"ShepardSteps"))
			{
				child ->QueryUnsignedAttribute("value", &h_exec_params_.density_renormalization_frequency);
			}

			if(!strcmp(key,"TimeMax"))
			{
				child ->QueryDoubleAttribute("value", &h_exec_params_.simulation_duration);
			}

			if(!strcmp(key,"TimeOut"))
			{
				child ->QueryDoubleAttribute("value", &h_exec_params_.save_interval);
			}
		}

		// particles
		child = pElem -> FirstChildElement("particles");
		child ->QueryUnsignedAttribute("np", &h_simulation_params_.num_particles);
		child ->QueryUnsignedAttribute("nb", &h_simulation_params_.num_boundary_particles);

		// constants
		child = pElem -> FirstChildElement("constants")->FirstChildElement();
		for(child;child;child = child -> NextSiblingElement())
		{

			key = child -> Value();

			if(!strcmp(key,"gravity"))
			{
				child ->QueryDoubleAttribute("x", &h_simulation_params_.gravity.x);
				child ->QueryDoubleAttribute("y", &h_simulation_params_.gravity.y);
				child ->QueryDoubleAttribute("z", &h_simulation_params_.gravity.z);
			}
			if(!strcmp(key,"cflnumber"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.cfl_number);
			}
			if(!strcmp(key,"gamma"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.gamma);
			}
			if(!strcmp(key,"rhop0"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.rhop0);
			}
			if(!strcmp(key,"eps"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.eps);

			}
			if(!strcmp(key,"dp"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.dp);

			}
			if(!strcmp(key,"h"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.smoothing_length);

			}
			if(!strcmp(key,"b"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.b_coeff);

			}
			if(!strcmp(key,"massbound"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.boundary_mass);

			}
			if(!strcmp(key,"massfluid"))
			{
				child ->QueryDoubleAttribute("value", &h_simulation_params_.fluid_mass);

			}
		}
	}
#else

	// block: definition
	{

		pElem = hRoot.FirstChild( "casedef" ).FirstChild( "geometry" ).FirstChild( "definition" ).Element();
		pElem = pElem -> FirstChildElement();
		for (pElem; pElem; pElem = pElem -> NextSiblingElement())
		{
			if (!strcmp("pointmin",pElem->Value()))
			{
				pElem ->QueryFloatAttribute("x", &h_domain_params_.world_origin.x);
				pElem ->QueryFloatAttribute("y", &h_domain_params_.world_origin.y);
				pElem ->QueryFloatAttribute("z", &h_domain_params_.world_origin.z);
			}
			else if (!strcmp("pointmax",pElem->Value()))
			{
				pElem ->QueryFloatAttribute("x", &h_domain_params_.world_limits.x);
				pElem ->QueryFloatAttribute("y", &h_domain_params_.world_limits.y);
				pElem ->QueryFloatAttribute("z", &h_domain_params_.world_limits.z);
			}

		}
	}

	// block: execution
	{
		TiXmlElement *child;
		const char *key;
		double value;
		pElem = hRoot.FirstChild( "execution").Element();

		// parameters
		child = pElem -> FirstChildElement("parameters")->FirstChildElement();
		for(child;child;child = child -> NextSiblingElement())
		{
			key = child ->Attribute("key");
			if(!strcmp(key,"Visco"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.nu);
			}

			if(!strcmp(key,"ShepardSteps"))
			{
				child ->QueryUnsignedAttribute("value", &h_exec_params_.density_renormalization_frequency);
			}

			if(!strcmp(key,"TimeMax"))
			{
				child ->QueryFloatAttribute("value", &h_exec_params_.simulation_duration);
			}

			if(!strcmp(key,"TimeOut"))
			{
				child ->QueryFloatAttribute("value", &h_exec_params_.save_interval);
			}
		}

		// particles
		child = pElem -> FirstChildElement("particles");
		child ->QueryUnsignedAttribute("np", &h_simulation_params_.num_particles);
		child ->QueryUnsignedAttribute("nb", &h_simulation_params_.num_boundary_particles);

		// constants
		child = pElem -> FirstChildElement("constants")->FirstChildElement();
		for(child;child;child = child -> NextSiblingElement())
		{

			key = child -> Value();

			if(!strcmp(key,"gravity"))
			{
				child ->QueryFloatAttribute("x", &h_simulation_params_.gravity.x);
				child ->QueryFloatAttribute("y", &h_simulation_params_.gravity.y);
				child ->QueryFloatAttribute("z", &h_simulation_params_.gravity.z);
			}
			if(!strcmp(key,"cflnumber"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.cfl_number);
			}
			if(!strcmp(key,"gamma"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.gamma);
			}
			if(!strcmp(key,"rhop0"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.rhop0);
			}
			if(!strcmp(key,"eps"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.eps);

			}
			if(!strcmp(key,"dp"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.dp);

			}
			if(!strcmp(key,"h"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.smoothing_length);

			}
			if(!strcmp(key,"b"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.b_coeff);

			}
			if(!strcmp(key,"massbound"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.boundary_mass);

			}
			if(!strcmp(key,"massfluid"))
			{
				child ->QueryFloatAttribute("value", &h_simulation_params_.fluid_mass);

			}
		}
	}
#endif
	setDerivedInputs();

} //end ParticleSystem::loadCfg()

void
ParticleSystem::setDerivedInputs()
{
	// Domain Parameters
	{
		h_domain_params_.domainMin = h_domain_params_.world_origin - 0.1 * h_simulation_params_.smoothing_length;
		h_domain_params_.domainMax = h_domain_params_.world_limits + 0.1 * h_simulation_params_.smoothing_length;

		h_domain_params_.world_size = h_domain_params_.domainMax - h_domain_params_.domainMin;

		Real cellSize = 2.0 * h_simulation_params_.smoothing_length;  // cell size equal to 2*particle smoothing length
		h_domain_params_.cell_size = make_Real3(cellSize, cellSize, cellSize); // uniform grid spacing

		h_domain_params_.grid_size.x = floor((h_domain_params_.domainMax.x - h_domain_params_.domainMin.x)/h_domain_params_.cell_size.x) + 1;
		h_domain_params_.grid_size.y = floor((h_domain_params_.domainMax.y - h_domain_params_.domainMin.y)/h_domain_params_.cell_size.y) + 1;
		h_domain_params_.grid_size.z = floor((h_domain_params_.domainMax.z - h_domain_params_.domainMin.z)/h_domain_params_.cell_size.z) + 1;

		h_domain_params_.num_cells = h_domain_params_.grid_size.x*h_domain_params_.grid_size.y*h_domain_params_.grid_size.z;
	}

	// Simulation Parameters
	{
		h_simulation_params_.num_fluid_particles = h_simulation_params_.num_particles - h_simulation_params_.num_boundary_particles;
		h_simulation_params_.epsilon = 0.1 * h_simulation_params_.smoothing_length;
		h_simulation_params_.epsilon = h_simulation_params_.epsilon * h_simulation_params_.epsilon;
		h_simulation_params_.pi = 3.14159265358979323846;
		h_simulation_params_.over_rhop0 = 1.0 / h_simulation_params_.rhop0;
		h_simulation_params_.over_smoothing_length = 1.0 / h_simulation_params_.smoothing_length;
		h_simulation_params_.four_h_squared = 4.0 * h_simulation_params_.smoothing_length * h_simulation_params_.smoothing_length;
		h_simulation_params_.cs0 = sqrt(h_simulation_params_.b_coeff * h_simulation_params_.gamma * h_simulation_params_.over_rhop0);
	}

    // Execution parameters
    h_exec_params_.fixed_dt = 0.0;
    h_exec_params_.periodic_in = NONE;
    h_exec_params_.print_interval = h_exec_params_.save_interval;

    if (h_simulation_params_.eps == 0.0)  h_exec_params_.xsph = false;
	if (h_domain_params_.world_origin.y == h_domain_params_.world_limits.y)
	{
		h_exec_params_.simulation_dimension = TWO_D;
	}
	else
	{
		h_exec_params_.simulation_dimension = THREE_D;
	}

	// Wendland smoothing kernel coefficients
    if(h_exec_params_.simulation_dimension == TWO_D)
    {
    	h_simulation_params_.wendland_a1 = 7.0/(4.0 * h_simulation_params_.pi * h_simulation_params_.smoothing_length * h_simulation_params_.smoothing_length);
    }
    else
    {
    	h_simulation_params_.wendland_a1 = 21.0/(16.0 * h_simulation_params_.pi * h_simulation_params_.smoothing_length * h_simulation_params_.smoothing_length * h_simulation_params_.smoothing_length);
    }

    h_simulation_params_.wendland_a2 = -5.0 * h_simulation_params_.over_smoothing_length * h_simulation_params_.wendland_a1;
    h_simulation_params_.sheppard = h_simulation_params_.wendland_a1;

    // Save parameters to gpu memory
    gpusph::set_domain_parameters(&h_domain_params_);
    gpusph::set_sim_parameters(&h_simulation_params_);
    gpusph::set_exec_parameters(&h_exec_params_);

}
void
ParticleSystem::load(std::string config)
{
	std::ifstream p_file(config.c_str());

	if(p_file.is_open())
	{
		Real x,y,z,u,v,w,rho_temp,p_temp;
		uint type_temp;

		for( uint i = 0; i < h_simulation_params_.num_particles; i++)
			{
				p_file >> x >> y >> z >> u >> v >> w >> rho_temp >> p_temp >> type_temp;

				h_pospres_[4*i] = x;
				h_pospres_[4*i+1] = y;
				h_pospres_[4*i+2] = z;
				h_pospres_[4*i+3] = p_temp;

				h_velrhop_[4*i] = u;
				h_velrhop_[4*i+1] = v;
				h_velrhop_[4*i+2] = w;
				h_velrhop_[4*i+3] = rho_temp;

				h_particle_type_[i] = type_temp;
			}

		gpusph::copyArrayToDevice(d_pospres_, h_pospres_, 0, numParticles_*4*sizeof(Real));
		gpusph::copyArrayToDevice(d_velrhop_, h_velrhop_, 0, numParticles_*4*sizeof(Real));
		gpusph::copyArrayToDevice(d_particle_type_, h_particle_type_, 0, numParticles_*sizeof(uint));
	}

	else std::cout << "Unable to open file containing initial particle distribution";

}

//Real
//ParticleSystem::get_time_step()
//{
//	cuda_vector_norm(d_ace_drho_dt_,d_norm_ace_);
//	Real max_acceleration = cuda_max(d_mod_ace_,numParticles_);
//	Real max_visc_dt = cuda_max(d_visc_dt_,numParticles_);
//	Real max_sound_speed = cuda_max();
//    Real smoothing_length = h_simulation_params_.smoothing_length;
//
//    //-dt1 depends on force per unit mass.
//	const Real dt1=(max_acceleration? (sqrt(smoothing_length)/sqrt(sqrt(max_acceleration))): FLT_MAX);
//
//	//-dt2 combines the Courant and the viscous time-step controls.
//    const Real dt2=(max_sound_speed||max_visc_dt? (smoothing_length/(max_sound_speed+smoothing_length*max_visc_dt)): FLT_MAX);
//
//    //-dt new value of time step.
//    Real dt=h_simulation_params_.cfl_number*min(dt1,dt2);
//    //if(DtFixed)dt=DtFixed->GetDt(TimeStep,dt);
//    //if(dt < dt_min){ dt = dt_min; }
//    return(dt);
//}
