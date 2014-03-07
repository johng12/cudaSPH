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
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654
#endif

ParticleSystem::ParticleSystem(uint numParticles) :
    initialized_(false),
    numParticles_(numParticles),
    h_pospres_(0),
    h_velrhop_(0),
    d_pospres_(0),
    d_velrhop_(0),
    d_pospres_pre_(0),
	d_velrhop_pre_(0),
	d_ace_drho_dt_(0),
	d_visc_dt_(0),
    m_timer(NULL),
    m_solverIterations(1)
{
	// set simulation parameters
	h_simulation_params_.num_particles = numParticles;
	h_simulation_params_.num_fluid_particles = numParticles;
	h_simulation_params_.num_boundary_particles = numParticles;
	h_simulation_params_.gravity = make_Real3(0.0, 0.0, -9.81);
	h_simulation_params_.smoothing_length = 1.0 / 64.0;
	h_simulation_params_.over_smoothing_length = 1.0 / h_simulation_params.smoothing_length;
	h_simulation_params_.four_h_squared = 4.0 * h_simulation_params_.smoothing_length * h_simulation_params_.smoothing_length;
	h_simulation_params_.rhop0 = 1000.0;
	h_simulation_params_.over_rhop0 = 1.0 / 1000.0;
	h_simulation_params_.fluid_mass = 1.0;
	h_simulation_params_.boundary_mass = 1.0;
	h_simulation_params_.cs0 = 1.0;
	h_simulation_params_.wendland_a1 = 1.0;
	h_simulation_params_.wendland_a2 = 1.0;
	h_simulation_params_.epsilon = 1e-3;
	h_simulation_params_.nu = 1e-3;
	h_simulation_params_.gamma = 7.0;
	h_simulation_params_.b_coeff = 1.0;
	h_simulation_params_.cfl_number = 0.3;

	// set domain parameters
	h_domain_params_.world_origin = make_Real3(-1.0, -1.0, -1.0);
    h_domain_params_.world_size = make_Real3(2.0, 2.0, 2.0);
    Real3 domainMin = h_domain_params_.world_origin - 2.0 * h_domain_params_.smoothingLength;
    h_domain_params_.world_origin = domainMin;
	Real3 domainMax = domainMin + h_domain_params_.world_size + 2.0 * h_domain_params_.smoothingLength;
    Real cellSize = 2.0 * h_domain_params_.smoothingLength;  // cell size equal to 2*particle smoothing length
	h_domain_params_.cell_size = make_Real3(cellSize, cellSize, cellSize); // uniform grid spacing

	h_grid_size_.x = floor((domainMax.x - domainMin.x)/h_domain_params_.cell_size.x);
	h_grid_size_.y = floor((domainMax.y - domainMin.y)/h_domain_params_.cell_size.y);
	h_grid_size_.z = floor((domainMax.z - domainMin.z)/h_domain_params_.cell_size.z);

//	m_gridSize.x = m_gridSize.y = m_gridSize.z = 64;
    h_numGridCells_ = h_grid_size_.x*h_grid_size_.y*h_grid_size_.z;
    h_domain_params_.num_cells = h_numGridCells_;

    // set execution parameters
    h_exec_params_.density_renormalization_frequency = 30;
    h_exec_params_.fixed_dt = 0.0;
    h_exec_params_.periodic_in = NONE;
    h_exec_params_.print_interval = 1e-3;
    h_exec_params_.save_interval = 1e-3;
    h_exec_params_.simulation_dimension = THREE_D;
    h_exec_params_.output_directory = "./Case_out/";
    h_exec_params_.working_directory = "./";

    _initialize(numParticles);

}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    numParticles_ = 0;
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!initialized_);

    numParticles_ = numParticles;

    // allocate host storage
    h_pospres_ = new Real[numParticles_*4];
    h_velrhop_ = new Real[numParticles_*4];
    memset(h_pospres_, 0, numParticles_*4*sizeof(Real));
    memset(h_velrhop_, 0, numParticles_*4*sizeof(Real));

    h_cell_start_ = new uint[h_numGridCells_];
    memset(h_cell_start_, 0, h_numGridCells_*sizeof(uint));

    h_particle_hash_ = new uint[numParticles_];
	memset(h_particle_hash_, 0, numParticles_*sizeof(uint));

	h_particle_index_ = new uint[numParticles_];
		memset(h_particle_index_, 0, numParticles_*sizeof(uint));

	h_particle_type_ = new uint[numParticles_];
			memset(h_particle_type_, 0, numParticles_*sizeof(uint));

    h_cell_end_ = new uint[h_numGridCells_];
    memset(h_cell_end_, 0, h_numGridCells_*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(Real) * 4 * numParticles_;

	allocateArray((void **)&d_pospres_, memSize);
    allocateArray((void **)&d_velrhop_, memSize);
    allocateArray((void **)&d_pospres_pre_, memSize);
	allocateArray((void **)&d_velrhop_pre_, memSize);
	allocateArray((void **)&d_ace_drho_dt_, memSize);
	allocateArray((void **)&d_particle_type_, sizeof(uint) * numParticles_);
    allocateArray((void **)&d_sorted_pospres_, memSize);
    allocateArray((void **)&d_sorted_velrhop_, memSize);
    allocateArray((void **)&d_sorted_type_, sizeof(uint) * numParticles_);

    allocateArray((void **)&d_visc_dt_,sizeof(Real) * numParticles_);
    allocateArray((void **)&d_max_accel_,sizeof(Real));
    allocateArray((void **)&d_max_sound_speed_,sizeof(Real));

    allocateArray((void **)&d_density_sum_,sizeof(Real) * numParticles_);
    allocateArray((void **)&d_kernel_sum_, sizeof(Real) * numParticles_);

    allocateArray((void **)&d_particle_hash_, numParticles_*sizeof(uint));
    allocateArray((void **)&d_particle_index_, numParticles_*sizeof(uint));
    allocateArray((void **)&d_cell_start_, h_numGridCells_*sizeof(uint));
    allocateArray((void **)&d_cell_end_, h_numGridCells_*sizeof(uint));

    sdkCreateTimer(&m_timer);

    setParameters(&h_domain_params_);
    setParameters(&h_simulation_params_);
    setParameters(&h_exec_params_);

    initialized_ = true;
}

void
ParticleSystem::_finalize()
{
    assert(initialized_);

    delete [] h_pospres_;
    delete [] h_velrhop_;
    delete [] h_cell_start_;
    delete [] h_cell_end_;

    freeArray(d_pospres_);
    freeArray(d_velrhop_);
    freeArray(d_pospres_pre_);
    freeArray(d_velrhop_pre_);
    freeArray(d_ace_drho_dt_);
    freeArray(d_particle_type_);
    freeArray(d_sorted_pospres_);
    freeArray(d_sorted_velrhop_);
    freeArray(d_sorted_type_);

    freeArray(d_visc_dt_);
    freeArray(d_max_accel_);
    freeArray(d_max_sound_speed_);

    freeArray(d_density_sum_);
    freeArray(d_kernel_sum_);

    freeArray(d_particle_hash_);
    freeArray(d_particle_index_);
    freeArray(d_cell_start_);
    freeArray(d_cell_end_);
}

// step the simulation and return the current time step
Real
ParticleSystem::update(Real deltaTime)
{
    assert(initialized_);

    // update constants
    setParameters(&h_domain_params_);

    {//======== Predictor Step =============
		// calculate grid hash
		calcHash(
			d_particle_hash_,
			d_particle_index_,
			d_pospres_pre_,
			numParticles_);

		// sort particles based on hash
		sortParticles(d_particle_hash_, d_particle_index_, numParticles_);

		// reorder particle arrays into sorted order and
		// find start and end of each cell
		reorderDataAndFindCellStart(
			d_cell_start_,
			d_cell_end_,
			d_sorted_pospres_,
			d_sorted_velrhop_,
			d_particle_hash_,
			d_particle_index_,
			d_pospres_pre_,
			d_velrhop_pre_,
			numParticles_,
			h_numGridCells_);

		// prepare variables for interaction
		// zero accel arrays, get pressures, etc.
		pre_interaction(d_ace_drho_dt_,
						d_velrhop_,
						d_pospres_,
						d_visc_dt_,
						numParticles_);

		// process particle interactions
		collide(
			d_velrhop_,
			d_sorted_pospres_,
			d_sorted_velrhop_,
			d_particle_index_,
			d_cell_start_,
			d_cell_end_,
			numParticles_,
			h_numGridCells_);

		// get time step
		deltaTime = get_time_step(d_visc_dt_,numParticles_);

		// zero out acceleration of stationary particles. still need to add C wrapper in particleSystem_cuda.cu for this
		zero_acceleration(d_ace_drho_dt_);

		// zero out y-component of data for 2D simulations
		if(h_exec_params_.simulation_dimension == TWO_D) zero_ycomponent(d_ace_drho_dt_,numParticles_);

		// predictor step
		predictorStep(
			d_pospres_,
			d_velrhop_,
			d_pospres_pre_,
			d_velrhop_pre_,
			deltaTime,
			numParticles_);
    }

    {//============ Corrector Step =============
		// calculate grid hash
		calcHash(
			d_particle_hash_,
			d_particle_index_,
			d_pospres_pre_,
			numParticles_);

		// sort particles based on hash
		sortParticles(d_particle_hash_, d_particle_index_, numParticles_);

		// reorder particle arrays into sorted order and
		// find start and end of each cell
		reorderDataAndFindCellStart(
			d_cell_start_,
			d_cell_end_,
			d_sorted_pospres_,
			d_sorted_velrhop_,
			d_particle_hash_,
			d_particle_index_,
			d_pospres_pre_,
			d_velrhop_pre_,
			numParticles_,
			h_numGridCells_);

		// prepare data for interactions
		pre_interaction(d_ace_drho_dt_,
								d_velrhop_,
								d_pospres_,
								d_visc_dt_,
								numParticles_);

		// process particle interactions
		collide(
			d_velrhop_,
			d_sorted_pospres_,
			d_sorted_velrhop_,
			d_particle_index_,
			d_cell_start_,
			d_cell_end_,
			numParticles_,
			h_numGridCells_);

		// zero out acceleration of stationary particles. still need to add C wrapper in particleSystem_cuda.cu for this
		zero_acceleration(d_ace_drho_dt_);

		// zero out y-component of data for 2D simulations. still need to add C wrapper to this as well
		if(h_exec_params_.simulation_dimension == TWO_D) zero_ycomponent(d_ace_drho_dt_,numParticles_);

		// corrector step
        correctorStep(
            d_pospres_,
            d_velrhop_,
            d_pospres_pre_,
            d_velrhop_pre_,
            deltaTime,
            numParticles_);
    }

    return deltaTime;

}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(h_cell_start_, d_cell_start_, 0, sizeof(uint)*h_numGridCells_);
    copyArrayFromDevice(h_cell_end_, d_cell_end_, 0, sizeof(uint)*h_numGridCells_);
    uint maxCellSize = 0;

    for (uint i=0; i<h_numGridCells_; i++)
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
    printf("world origin = %3.2f x %3.2f x %3.2f \n",h_domain_params_.world_origin.x,h_domain_params_.worldOrigin.y,h_domain_params_.world_origin.z);
    printf("world size = %3.2f x %3.2f x %3.2f \n",h_domain_params_.world_size.x,h_domain_params_.world_size.y,h_domain_params_.world_size.z);
    printf("Cell Size = %5.4f x %5.4f x %5.4f \n",h_domain_params_.cell_size.x, h_domain_params_.cell_size.y,h_domain_params_.cell_size.z);
    printf("Grid Size = %d x %d x %d \n",h_grid_size_.x, h_grid_size_.y,h_grid_size_.z);
    printf("Total Cells = %d \n",h_domain_params_.num_cells);
}

void
ParticleSystem::dumpParticles(uint start, uint count, const char *fileName)
{
	  FILE * pFile;
	  pFile = fopen (fileName,"w");

    // debug
    copyArrayFromDevice(h_pospres_, d_pospres_, 0, sizeof(Real)*4*count);
    copyArrayFromDevice(h_velrhop_, d_velrhop_, 0, sizeof(Real)*4*count);

    for (uint i=start; i<start+count; i++)  {
        //        printf("%d: ", i);
    	fprintf(pFile,"%10.9f, %10.9f, %10.9f\n", h_pospres_[i*4+0], h_pospres_[i*4+1], h_pospres_[i*4+2]);
//        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", h_pospres_[i*4+0], h_pospres_[i*4+1], h_pospres_[i*4+2], h_pospres_[i*4+3]);
//        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", h_velrhop_[i*4+0], h_velrhop_[i*4+1], h_velrhop_[i*4+2], h_velrhop_[i*4+3]);
    }
    fclose(pFile);
}

Real *
ParticleSystem::getArray(ParticleArray array)
{
    assert(initialized_);


	Real *hdata = 0;
	Real *ddata = 0;

    switch (array)
    {
        default:
        case POSITION:
            hdata = h_pospres_;
            ddata = d_pospres_;
            copyArrayFromDevice(hdata, ddata, 0,numParticles_*4*sizeof(Real));
            break;

        case VELOCITY:
            hdata = h_velrhop_;
            ddata = d_velrhop_;
            copyArrayFromDevice(hdata, ddata, 0,numParticles_*4*sizeof(Real));
            break;

    }

    return hdata;

}

uint *
ParticleSystem::getHash()
{
    assert(initialized_);

	uint *hdata = 0;
	uint *ddata = 0;

    hdata = h_particle_hash_;
    ddata = d_particle_hash_;
    copyArrayFromDevice(hdata,ddata,0,numParticles_*sizeof(uint));

    return hdata;

}

uint *
ParticleSystem::getIndex()
{
    assert(initialized_);

	uint *hdata = 0;
	uint *ddata = 0;

    hdata = h_particle_index_;
    ddata = d_particle_index_;
    copyArrayFromDevice(hdata,ddata,0,numParticles_*sizeof(uint));

    return hdata;

}

void
ParticleSystem::setArray(ParticleArray array, const Real *data, int start, int count)
{
    assert(initialized_);

    switch (array)
    {
        default:
        case POSITION:
			copyArrayToDevice(d_pospres_, data, start*4*sizeof(Real), count*4*sizeof(Real));
			break;



        case VELOCITY:
            copyArrayToDevice(d_velrhop_, data, start*4*sizeof(Real), count*4*sizeof(Real));
            break;
    }
}

inline Real frand()
{
    return rand() / (Real) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, Real spacing, Real jitter, uint numParticles)
{
    srand(1973);

    for (uint z=0; z<size[2]; z++)
    {
        for (uint y=0; y<size[1]; y++)
        {
            for (uint x=0; x<size[0]; x++)
            {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

                if (i < numParticles)
                {
                    h_pospres_[i*4] = (spacing * x) + h_domain_params_.particleRadius - 1.0 + (frand()*2.0-1.0)*jitter;
                    h_pospres_[i*4+1] = (spacing * y) + h_domain_params_.particleRadius - 1.0 + (frand()*2.0-1.0)*jitter;
                    h_pospres_[i*4+2] = (spacing * z) + h_domain_params_.particleRadius - 1.0 + (frand()*2.0-1.0)*jitter;

//                    h_pospres_[i*4] = (spacing * x) + m_params.particleRadius - 1.0;
//				    h_pospres_[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0;
//				    h_pospres_[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0;
                    h_pospres_[i*4+3] = 1.0;

                    h_velrhop_[i*4] = 0.0;
                    h_velrhop_[i*4+1] = 0.0;
                    h_velrhop_[i*4+2] = 0.0;
                    h_velrhop_[i*4+3] = 0.0;
                }
            }
        }
    }
}

void
ParticleSystem::reset(ParticleConfig config)
{
    switch (config)
    {
        default:
        case CONFIG_RANDOM:
            {
                int p = 0, v = 0;

                for (uint i=0; i < numParticles_; i++)
                {
                    Real point[3];
                    point[0] = frand();
                    point[1] = frand();
                    point[2] = frand();
                    h_pospres_[p++] = 2 * (point[0] - 0.5);
                    h_pospres_[p++] = 2 * (point[1] - 0.5);
                    h_pospres_[p++] = 2 * (point[2] - 0.5);
                    h_pospres_[p++] = 1.0; // radius
                    h_velrhop_[v++] = 0.0;
                    h_velrhop_[v++] = 0.0;
                    h_velrhop_[v++] = 0.0;
                    h_velrhop_[v++] = 0.0;
                }
            }
            break;

        case CONFIG_GRID:
            {
                Real jitter = h_domain_params_.particleRadius*0.01;
                uint s = (int) ceil(pow((Real) numParticles_, 1.0 / 3.0));
                uint gridSize[3];
                gridSize[0] = gridSize[1] = gridSize[2] = s;
                initGrid(gridSize, h_domain_params_.particleRadius*2.0, jitter, numParticles_);
            }
            break;
    }

    setArray(POSITION, h_pospres_, 0, numParticles_);
    setArray(VELOCITY, h_velrhop_, 0, numParticles_);
}
