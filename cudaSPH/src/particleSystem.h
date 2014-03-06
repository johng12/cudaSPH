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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
            DENSITY,
            HASH
        };

        Real update(Real deltaTime);
        void reset(ParticleConfig config);

        Real *getArray(ParticleArray array);
        uint *getHash();
        uint *getIndex();
        void   setArray(ParticleArray array, const Real *data, int start, int count);

        int    getNumParticles() const
        {
            return numParticles_;
        }

        void dumpGrid();
        void dumpParticles(uint start, uint count, const char *fileName);
        void dumpParameters();

        void setIterations(int i)
        {
            m_solverIterations = i;
        }

        void setDamping(Real x)
        {
            h_domain_params_.globalDamping = x;
        }
        void setGravity(Real x)
        {
            h_domain_params_.gravity = make_Real3(0.0, x, 0.0);
        }

        void setCollideSpring(Real x)
        {
            h_domain_params_.spring = x;
        }
        void setCollideDamping(Real x)
        {
            h_domain_params_.damping = x;
        }
        void setCollideShear(Real x)
        {
            h_domain_params_.shear = x;
        }
        void setCollideAttraction(Real x)
        {
            h_domain_params_.attraction = x;
        }

        void setColliderPos(Real3 x)
        {
            h_domain_params_.colliderPos = x;
        }

        Real getParticleRadius()
        {
            return h_domain_params_.particleRadius;
        }
        Real3 getColliderPos()
        {
            return h_domain_params_.colliderPos;
        }
        Real getColliderRadius()
        {
            return h_domain_params_.colliderRadius;
        }
        uint3 getGridSize()
        {
            return h_domain_params_.grid_size;
        }
        Real3 getWorldOrigin()
        {
            return h_domain_params_.world_origin;
        }
        Real3 getCellSize()
        {
            return h_domain_params_.cell_size;
        }

    protected: // methods
        ParticleSystem() {}

        void _initialize(int numParticles);
        void _finalize();

        void initGrid(uint *size, Real spacing, Real jitter, uint numParticles);

    protected: // data
        bool m_bInitialized;
        uint numParticles_;

        // CPU particle data
        Real *h_pospres_;              // particle positions and pressures
        Real *h_velrhop_;              // particle velocities and densities
        uint *h_particle_type_; // number identifying particle type (e.g. fluid, stationary boundary, moving boundary, etc.)

        // CPU grid sort data - used for dumping grid info to file
        uint  *h_particle_hash_; // bin number that particle is located in for grid search
        uint  *h_particle_index_; // unique particle identifier. constant throughout simulation
        uint  *h_cell_start_; // sorted index of first particle in a given grid cell
        uint  *h_cell_end_; // sorted index of last particle in a given grid cell

        // GPU particle data
        Real *d_pospres_; // particle positions and pressures (x,y,z,P)
        Real *d_velrhop_; // particle velocities and densities (u,v,w,rho)
        Real *d_pospres_pre_; // stores position and pressure data after predictor step
        Real *d_velrhop_pre_; // stores velocity and density data after predictor step
        Real *d_ace_drho_dt_; // particle acceleration data (u_dot, v_dot, w_dot,rho_dot)
        uint *d_particle_type_; // particle type (fluid,boundary, etc.)
        Real *d_sorted_pospres_; // stores particle position and pressure data, sorted according to grid index
		Real *d_sorted_velrhop_; // stores particle velocity and density data, sorted according to grid index
		Real *d_sorted_type_; // stores particle type data, sorted according to grid index

        // Adaptive time step data
        Real *d_visc_dt_; // holds maximum dt value of each particle based on viscous considerations. (See Lui and Lui, 2003)
        Real *d_max_accel_; // holds maximum acceleration of all particles
        Real *d_max_sound_speed_; // holds maximum sound speed of all particles

        // Sheppard filter data
        Real *d_density_sum_; // density summation for each particle (rho(i) = Sum[ rho(j) * W_ij * Vol(j) ] for j = 1,2,...,N.
        Real *d_kernel_sum_; // kernel summation for each particle (Sum[ W_ij * Vol(j)] for j = 1,2,...,N.

        // grid data for sorting method
        uint  *d_particle_hash_; // grid hash value for each particle
        uint  *d_particle_index_;// particle index for each particle
        uint  *d_cell_start_;        // index of start of each cell in sorted list
        uint  *d_cell_end_;          // index of end of cell

        uint   m_gridSortBits; // A left over from the cuda particles code - don't know what this is for.

        // parameters
        simulation_parameters h_simulation_params_;
        domain_parameters h_domain_params_;
        execution_parameters h_exec_params_;

        uint3 h_grid_size_;
        uint h_numGridCells_;

        StopWatchInterface *m_timer;

        uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
