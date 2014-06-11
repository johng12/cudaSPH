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


ParticleSystem::ParticleSystem(uint numParticles):

    initialized_(false),
    numParticles_(numParticles),

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

	d_visc_dt_(0),
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
	// set simulation parameters
	h_simulation_params_.num_particles = 5281;
	h_simulation_params_.num_boundary_particles = 481;
	h_simulation_params_.num_fluid_particles = 4800;

	h_simulation_params_.gravity = make_Real3(0.0, 0.0, -9.81);
	h_simulation_params_.smoothing_length = 6.1237215018E-03;
	h_simulation_params_.over_smoothing_length = 1.0 / h_simulation_params_.smoothing_length;
	h_simulation_params_.four_h_squared = 4.0 * h_simulation_params_.smoothing_length * h_simulation_params_.smoothing_length;
	h_simulation_params_.rhop0 = 1000.0;
	h_simulation_params_.over_rhop0 = 1.0 / 1000.0;
	h_simulation_params_.fluid_mass = 2.5000000000E-02;
	h_simulation_params_.boundary_mass = 2.5000000000E-02;

	h_simulation_params_.wendland_a1 = 0.557/(h_simulation_params_.smoothing_length * h_simulation_params_.smoothing_length);
	h_simulation_params_.wendland_a2 = -2.7852/(h_simulation_params_.smoothing_length * h_simulation_params_.smoothing_length * h_simulation_params_.smoothing_length);
	h_simulation_params_.epsilon = 1e-3;
	h_simulation_params_.nu = 1e-3;
	h_simulation_params_.gamma = 7.0;
	h_simulation_params_.b_coeff = 1.6536857143E+05;
	h_simulation_params_.cs0 = sqrt(h_simulation_params_.b_coeff * h_simulation_params_.gamma * h_simulation_params_.over_rhop0);
	h_simulation_params_.cfl_number = 0.3;

	// set domain parameters
	h_domain_params_.world_origin = make_Real3(0.0, 0.0, 0.0);
    h_domain_params_.world_size = make_Real3(1.6, 0.67, 0.4);
    Real3 domainMin = h_domain_params_.world_origin - 0.1 * h_simulation_params_.smoothing_length;
	Real3 domainMax = h_domain_params_.world_origin + h_domain_params_.world_size + 0.1 * h_simulation_params_.smoothing_length;
	h_domain_params_.world_origin = domainMin;

	printf("%9.8f %9.8f %9.8f \n",domainMax.x,domainMax.y,domainMax.z);

    Real cellSize = 2.0 * h_simulation_params_.smoothing_length;  // cell size equal to 2*particle smoothing length
	h_domain_params_.cell_size = make_Real3(cellSize, cellSize, cellSize); // uniform grid spacing

	h_grid_size_.x = floor((domainMax.x - domainMin.x)/h_domain_params_.cell_size.x);
	h_grid_size_.y = floor((domainMax.y - domainMin.y)/h_domain_params_.cell_size.y);
	h_grid_size_.z = floor((domainMax.z - domainMin.z)/h_domain_params_.cell_size.z);
	h_domain_params_.grid_size.x = h_grid_size_.x;
	h_domain_params_.grid_size.y = h_grid_size_.y;
	h_domain_params_.grid_size.z = h_grid_size_.z;

    h_numGridCells_ = h_grid_size_.x*h_grid_size_.y*h_grid_size_.z;
    h_domain_params_.num_cells = h_numGridCells_;

    // set execution parameters
    h_exec_params_.density_renormalization_frequency = 30;
    h_exec_params_.fixed_dt = 0.0;
    h_exec_params_.periodic_in = NONE;
    h_exec_params_.print_interval = 1e-3;
    h_exec_params_.save_interval = 1e-3;
    h_exec_params_.simulation_dimension = TWO_D;

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

    h_particle_type_ = new uint[numParticles_];
		memset(h_particle_type_, 0, numParticles_*sizeof(uint));

	h_particle_hash_ = new uint[numParticles_];
		memset(h_particle_hash_, 0, numParticles_*sizeof(uint));

	h_particle_gridPos_ = new int[numParticles_*4];
		memset(h_particle_gridPos_,0,numParticles_*4*sizeof(int));

	h_particle_index_ = new uint[numParticles_];
		memset(h_particle_index_, 0, numParticles_*sizeof(uint));

    h_cell_start_ = new uint[h_numGridCells_];
    	memset(h_cell_start_, 0, h_numGridCells_*sizeof(uint));

    h_cell_end_ = new uint[h_numGridCells_];
    	memset(h_cell_end_, 0, h_numGridCells_*sizeof(uint));

	h_neighbors_ = new uint[numParticles_];
		memset(h_neighbors_,0,numParticles*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(Real) * 4 * numParticles_;

	gpusph::allocateArray((void **)&d_pospres_, memSize);
	gpusph::allocateArray((void **)&d_velrhop_, memSize);
	gpusph::allocateArray((void **)&d_pospres_pre_, memSize);
	gpusph::allocateArray((void **)&d_velrhop_pre_, memSize);
	gpusph::allocateArray((void **)&d_ace_drho_dt_, memSize);
	gpusph::allocateArray((void **)&d_particle_type_, sizeof(uint) * numParticles_);
	gpusph::allocateArray((void **)&d_sorted_pospres_, memSize);
	gpusph::allocateArray((void **)&d_sorted_velrhop_, memSize);
	gpusph::allocateArray((void **)&d_sorted_type_, sizeof(uint) * numParticles_);

	gpusph::allocateArray((void **)&d_visc_dt_,sizeof(Real) * numParticles_);
	gpusph::allocateArray((void **)&d_norm_ace_,sizeof(Real) * numParticles_);
	gpusph::allocateArray((void **)&d_max_accel_,sizeof(Real));
	gpusph::allocateArray((void **)&d_max_sound_speed_,sizeof(Real));

	gpusph::allocateArray((void **)&d_density_sum_,sizeof(Real) * numParticles_);
	gpusph::allocateArray((void **)&d_kernel_sum_, sizeof(Real) * numParticles_);

	gpusph::allocateArray((void **)&d_particle_hash_, numParticles_*sizeof(uint));
	gpusph::allocateArray((void **)&d_particle_index_, numParticles_*sizeof(uint));
	gpusph::allocateArray((void **)&d_particle_gridPos_,numParticles_*4*sizeof(int));
	gpusph::allocateArray((void **)&d_cell_start_, h_numGridCells_*sizeof(uint));
	gpusph::allocateArray((void **)&d_cell_end_, h_numGridCells_*sizeof(uint));
	gpusph::allocateArray((void **)&d_neighbors_,sizeof(uint) * numParticles_);
    sdkCreateTimer(&m_timer);

    gpusph::set_domain_parameters(&h_domain_params_);
    gpusph::set_sim_parameters(&h_simulation_params_);
    gpusph::set_exec_parameters(&h_exec_params_);

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

    gpusph::freeArray(d_visc_dt_);
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
ParticleSystem::update(Real deltaTime)
{
    assert(initialized_);

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
			numParticles_,
			d_particle_gridPos_);

//		gpusph::copyArrayFromDevice(h_particle_hash_,d_particle_hash_,0,sizeof(uint)*numParticles_);
//		gpusph::copyArrayFromDevice(h_particle_gridPos_,d_particle_gridPos_,0,sizeof(int)*4*numParticles_);
//		FILE *pFile;
//		pFile = fopen ("hash_data.dat","w");
//		for (uint i=0; i<numParticles_; i++)
//		{
//			fprintf(pFile,"%d %d %d %d\n", h_particle_gridPos_[i*4+0],h_particle_gridPos_[i*4+1],h_particle_gridPos_[i*4+2], h_particle_hash_[i]);
//		}
//		fclose(pFile);

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
			h_numGridCells_);

//		gpusph::copyArrayFromDevice(h_cell_start_, d_cell_start_, 0, sizeof(uint)*h_numGridCells_);
//		gpusph::copyArrayFromDevice(h_cell_end_,d_cell_end_,0,sizeof(uint)*h_numGridCells_);
//
//		pFile = fopen ("cell_Data.dat","w");
//		for (uint i=0; i<h_numGridCells_; i++)
//		{
//			fprintf(pFile,"%d %d\n", h_cell_start_[i],h_cell_end_[i]);
//		}
//		fclose(pFile);

		// prepare variables for interaction
		// zero accel arrays, get pressures, etc.
		// prepare data for interactions
		gpusph::pre_interaction(d_ace_drho_dt_,
						d_sorted_velrhop_,
						d_sorted_pospres_,
						d_visc_dt_,
						numParticles_);


		// process particle interactions
		gpusph::compute_interactions(d_ace_drho_dt_,
							 d_sorted_velrhop_,
							 d_sorted_pospres_,
							 d_particle_index_,
							 d_cell_start_,
							 d_cell_end_,
							 d_sorted_type_,
							 d_visc_dt_,
							 numParticles_,
							 h_numGridCells_,
							 d_neighbors_);
//
//		// get time step
//		deltaTime = get_time_step(d_visc_dt_,numParticles_);

		// zero out acceleration of stationary particles.
		gpusph::zero_acceleration(d_ace_drho_dt_,numParticles_);

		// zero out y-component of data for 2D simulations
		if(h_exec_params_.simulation_dimension == TWO_D) gpusph::zero_ycomponent(d_ace_drho_dt_,numParticles_);

//		gpusph::copyArrayFromDevice(h_pospres_, d_ace_drho_dt_, 0, sizeof(Real)*4*numParticles_);
//		FILE *pFile;
//		pFile = fopen ("ace_Data.dat","w");
//		for (uint i=0; i<numParticles_; i++)
//		{
//			fprintf(pFile,"%10.9e %10.9e %10.9e %10.9e\n", h_pospres_[i*4+0],h_pospres_[i*4+1],h_pospres_[i*4+2],h_pospres_[i*4+3]);
//		}
//		fclose(pFile);

		// predictor step
		gpusph::predictorStep(
			d_pospres_,
			d_velrhop_,
			d_pospres_pre_,
			d_velrhop_pre_,
			d_ace_drho_dt_,
			deltaTime,
			numParticles_);
    }

    {//============ Corrector Step =============
		// calculate grid hash
		gpusph::calcHash(
			d_particle_hash_,
			d_particle_index_,
			d_pospres_pre_,
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
			d_pospres_pre_,
			d_velrhop_pre_,
			d_particle_type_,
			numParticles_,
			h_numGridCells_);


		// prepare data for interactions
		gpusph::pre_interaction(d_ace_drho_dt_,
						d_sorted_velrhop_,
						d_sorted_pospres_,
						d_visc_dt_,
						numParticles_);

		// process particle interactions
		gpusph::compute_interactions(d_ace_drho_dt_,
    						 d_sorted_velrhop_,
							 d_sorted_pospres_,
							 d_particle_index_,
							 d_cell_start_,
							 d_cell_end_,
							 d_sorted_type_,
							 d_visc_dt_,
							 numParticles_,
							 h_numGridCells_,
							 d_neighbors_);

		// zero out acceleration of stationary particles. still need to add C wrapper in particleSystem_cuda.cu for this
		gpusph::zero_acceleration(d_ace_drho_dt_,numParticles_);

		// zero out y-component of data for 2D simulations. still need to add C wrapper to this as well
		if(h_exec_params_.simulation_dimension == TWO_D) gpusph::zero_ycomponent(d_ace_drho_dt_,numParticles_);

		// corrector step
        gpusph::correctorStep(
            d_pospres_,
            d_velrhop_,
            d_pospres_pre_,
            d_velrhop_pre_,
            d_ace_drho_dt_,
            deltaTime,
            numParticles_);

		// evaluate new pressures.
		gpusph::pre_interaction(d_ace_drho_dt_,
						d_velrhop_,
						d_pospres_,
						d_visc_dt_,
						numParticles_);
    }

    return deltaTime;


}

void
ParticleSystem::apply_shepard_filter()
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
		h_numGridCells_);

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
									d_sorted_velrhop_,
									d_sorted_pospres_,
									d_particle_index_,
									d_cell_start_,
									d_cell_end_,
									d_sorted_type_,
									numParticles_,
									h_numGridCells_);

}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
	gpusph::copyArrayFromDevice(h_cell_start_, d_cell_start_, 0, sizeof(uint)*h_numGridCells_);
	gpusph::copyArrayFromDevice(h_cell_end_, d_cell_end_, 0, sizeof(uint)*h_numGridCells_);
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
    printf("world origin = %3.2f x %3.2f x %3.2f \n",h_domain_params_.world_origin.x,h_domain_params_.world_origin.y,h_domain_params_.world_origin.z);
    printf("world size = %3.2f x %3.2f x %3.2f \n",h_domain_params_.world_size.x,h_domain_params_.world_size.y,h_domain_params_.world_size.z);
    printf("Cell Size = %5.4f x %5.4f x %5.4f \n",h_domain_params_.cell_size.x, h_domain_params_.cell_size.y,h_domain_params_.cell_size.z);
    printf("Grid Size = %d x %d x %d \n",h_grid_size_.x, h_grid_size_.y,h_grid_size_.z);
    printf("Total Cells = %d \n",h_domain_params_.num_cells);
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
