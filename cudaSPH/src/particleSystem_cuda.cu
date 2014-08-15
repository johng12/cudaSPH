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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include <helper_functions.h>
#include "particles_kernel.cuh"

#if USE_TEX
// textures for particle position and velocity
texture<Real4, 1, cudaReadModeElementType> oldPosTex;
texture<Real4, 1, cudaReadModeElementType> oldVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ domain_parameters domain_params;
__constant__ simulation_parameters sim_params;
__constant__ execution_parameters exec_params;

namespace gpusph
{
    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

    void copyArrayFromDevice(void *host, const void *device,int offset, int size)
    {

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

    }

    void set_sim_parameters(simulation_parameters *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(sim_params, hostParams, sizeof(simulation_parameters)));
    }

    void set_domain_parameters(domain_parameters *hostParams)
	{
//    	printf("world origin = %3.2f x %3.2f x %3.2f \n",hostParams.world_origin.x,hostParams.world_origin.y,hostParams.world_origin.z);
//    	printf("world origin = %3.2f x %3.2f x %3.2f \n",hostParams->world_origin.x,hostParams->world_origin.y,hostParams->world_origin.z);
		// copy parameters to constant memory
		checkCudaErrors(cudaMemcpyToSymbol(domain_params, hostParams, sizeof(domain_parameters)));
	}

    void set_exec_parameters(execution_parameters *hostParams)
	{
		// copy parameters to constant memory
		checkCudaErrors(cudaMemcpyToSymbol(exec_params, hostParams, sizeof(execution_parameters)));
	}

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void predictorStep(Real *pospres,
                         Real *velrhop,
                         Real *pospres_pre,
                         Real *velrhop_pre,
                         Real *velxcor,
                         Real *ace_drhodt,
                         uint numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);
        predictorStepD<<< numBlocks, numThreads >>>((Real4 *) pospres,
        											(Real4 *) velrhop,
        											(Real4 *) pospres_pre,
        											(Real4 *) velrhop_pre,
        											(Real4 *) velxcor,
        											(Real4 *) ace_drhodt,
        											numParticles);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
    }

    void correctorStep(Real *pospres,
                         Real *velrhop,
                         Real *pospres_pre,
                         Real *velrhop_pre,
                         Real *ace_drhodt,
                         uint numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);
        correctorStepD<<< numBlocks, numThreads >>>((Real4 *) pospres,
        											(Real4 *) velrhop,
        											(Real4 *) pospres_pre,
        											(Real4 *) velrhop_pre,
        											(Real4 *) ace_drhodt,
        											numParticles);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
    }

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  Real *pos,
                  int    numParticles,
                  int   *gridParticlePos)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (Real4 *) pos,
                                               numParticles,
                                               (int4 *) gridParticlePos);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

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
                                     uint  numCells)

    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(Real4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(Real4)));
#endif

        uint smemSize = sizeof(uint)*(numThreads+1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            (Real4 *) sortedPos,
            (Real4 *) sortedVel,
            sortedType,
            gridParticleHash,
            gridParticleIndex,
            (Real4 *) oldPos,
            (Real4 *) oldVel,
            oldType,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
    }

    void pre_interaction(Real *ace_drhodt, // output: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
			   Real *velrhop, // input: sorted velocity and density (v.x,v.y,v.z,rhop)
			   Real *pospres, // input: sorted particle positions and pressures
			   Real *viscdt, // output: max time step for adaptive time stepping
			   uint numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        pre_interactionD<<< numBlocks, numThreads >>>((Real4 *) ace_drhodt,
                                               (Real4 *) velrhop,
                                               (Real4 *) pospres,
                                               viscdt,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void compute_interactions(Real *ace_drhodt,
    			 Real *velxcorr,
                 Real *sorted_velrhop,
                 Real *sorted_pospres,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   *sorted_type,
                 Real  *viscdt,
                 uint   numParticles,
                 uint   numCells,
                 uint  *neighbors,
                 uint *gridParticleHash)
    {
#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, sorted_pospres, numParticles*sizeof(Real4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, sorted_velrhop, numParticles*sizeof(Real4)));
        checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
        checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        compute_particle_interactions<<< numBlocks, numThreads >>>((Real4 *) ace_drhodt,
																   (Real4 *) sorted_velrhop,
																   (Real4 *) sorted_pospres,
																   (Real4 *) velxcorr,
																   gridParticleIndex,
																   cellStart,
																   cellEnd,
																   sorted_type,
																   viscdt,
																   numParticles,
																   neighbors,
																   gridParticleHash);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
        checkCudaErrors(cudaUnbindTexture(cellStartTex));
        checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
    }


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::stable_sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }

    void reduceAccel(Real *ace_drhodt, // input: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
    				 Real *velrhop,
    				   Real *forcedt, // output: max time step for adaptive time stepping based on acceleration
    				   Real *viscdt, // output: max time step based on viscous diffusion
    				   uint numParticles)
    {
    	// thread per particle
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		reduceAccelD<<< numBlocks, numThreads >>>((Real4 *) ace_drhodt, // input: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
												  (Real4 *) velrhop,
		    				   	   	   	   	   	   forcedt, // output: max time step for adaptive time stepping based on acceleration
												   viscdt, // output: max time step based on viscous diffusion
												   numParticles);
    }

    Real get_time_step(Real *forcedt, Real *viscdt, uint numParticles)
    {
    	Real deltaTime = 0.0;
//    	Real *dt_cv = thrust::min_element(thrust::device_prt<double>(viscdt), thrust::device_prt<double>(viscdt) + numParticles);
//    	Real *dt_f = thrust::min_element(thrust::device_prt<double>(forcedt), thrust::device_prt<double>(forcedt) + numParticles);
//    	Real deltaTime = 0.3 * std::min(dt_cv,dt_f);
    	return deltaTime;
    }

    void zero_acceleration(Real *ace_drhodt, uint numParticles)
    {
    	// thread per particle
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

    	zero_accelerationD<<< numBlocks,numThreads >>>( (Real4 *) ace_drhodt);

    	// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
    }

    void zero_ycomponent(Real *data, uint numParticles)
    {
    	// thread per particle
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

    	zero_ycomponentD<<< numBlocks,numThreads >>>( (Real4 *) data, numParticles);

    	// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
    }
    void sheppard_density_filter(Real *velrhop,
    								Real *pospres,
    								 Real *sorted_velrhop,
    								 Real *sorted_pospres,
    								 uint *gridParticleIndex, // input: sorted particle indicies
    								 uint *cellStart,
    								 uint *cellEnd,
    								 uint  *sorted_type, // input: sorted particle type (e.g. fluid, boundary, etc.)
    								 uint numParticles)
    {

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // execute the kernel
    sheppard_density_filterD<<< numBlocks, numThreads >>>((Real4 *) velrhop,
    													   (Real4 *) pospres,
														   (Real4 *) sorted_velrhop,
														   (Real4 *) sorted_pospres,
														   gridParticleIndex,
														   cellStart,
														   cellEnd,
														   sorted_type,
														   numParticles);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
    }

// Begin CUDA kernel code:

    // calculate position in uniform grid
    __device__ int3 calcGridPos(Real3 p)
    {
        int3 gridPos;
        gridPos.x = floor((p.x - domain_params.domainMin.x) / domain_params.cell_size.x);
        gridPos.y = floor((p.y - domain_params.domainMin.y) / domain_params.cell_size.y);
        gridPos.z = floor((p.z - domain_params.domainMin.z) / domain_params.cell_size.z);
        if(gridPos.x >= domain_params.grid_size.x) gridPos.x = domain_params.grid_size.x - 1;
        if(gridPos.y >= domain_params.grid_size.y) gridPos.y = domain_params.grid_size.y - 1;
        if(gridPos.z >= domain_params.grid_size.z) gridPos.z = domain_params.grid_size.z - 1;
        return gridPos;
    }

    // calculate address in grid from position
    __device__ uint calcGridHash(int3 gridPos)
    {

    	return __umul24(__umul24(gridPos.z,domain_params.grid_size.y),domain_params.grid_size.x)
    			+ __umul24(gridPos.y,domain_params.grid_size.x) + gridPos.x;
    }

    // calculate grid hash value for each particle
    __global__
    void calcHashD(uint   *gridParticleHash,  // output
                   uint   *gridParticleIndex, // output
                   Real4 *pos,               // input: positions
                   uint    numParticles,
                   int4	  *gridParticlePos)
    {
        uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

        if (index >= numParticles) return;

        volatile Real4 p = pos[index];

        // get address in grid
        int3 gridPos = calcGridPos(make_Real3(p.x, p.y, p.z));

        uint hash = calcGridHash(gridPos);
        //printf("%d %d %d %d %d %d %d \n",gridPos.x,gridPos.y,gridPos.z,domain_params.grid_size.x,domain_params.grid_size.y,domain_params.grid_size.z,hash);
        // store grid hash and particle index
        gridParticleHash[index] = hash;
        gridParticleIndex[index] = index;
        gridParticlePos[index].x = gridPos.x; gridParticlePos[index].y = gridPos.y; gridParticlePos[index].z = gridPos.z;
    }

    // rearrange particle data into sorted order, and find the start of each cell
    // in the sorted hash array
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
                                      uint    numParticles)
    {
        extern __shared__ uint sharedHash[];    // blockSize + 1 elements
        uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

        uint hash;

        // handle case when no. of particles not multiple of block size
        if (index < numParticles)
        {
            hash = gridParticleHash[index];

            // Load hash data into shared memory so that we can look
            // at neighboring particle's hash value without loading
            // two hash values per thread
            sharedHash[threadIdx.x+1] = hash;

            if (index && !threadIdx.x)
            {
                // first thread in block must load neighbor particle hash
                sharedHash[0] = gridParticleHash[index-1];
            }
        }

        __syncthreads();

          if(index < numParticles){
            if(!index || hash != sharedHash[threadIdx.x]){
              cellStart[hash] = index;
              if(index)cellEnd[sharedHash[threadIdx.x]] = index;
            }
            if(index == numParticles - 1) cellEnd[hash] = index + 1;

            // Now use the sorted index to reorder the pos and vel data
            uint originalIndex = gridParticleIndex[index];
            Real4 pos = oldPos[originalIndex];
            Real4 vel = oldVel[originalIndex];
            uint p_type = oldType[originalIndex];

            sortedPos[index] = pos;
            sortedVel[index] = vel;
            sortedType[index] = p_type;
        }


    }

    __device__
    void particle_particle_interaction(Real4 pospres1, Real4 velrhop1, Real massp1,
    								   Real4 pospres2, Real4 velrhop2, Real massp2,
    								   Real3 &acep1, Real &arp1, Real &visc, uint &neigh1, Real3 &vcor, uint j_index)
    {

    	Real drx = pospres1.x - pospres2.x;
    	Real dry = pospres1.y - pospres2.y;
    	Real drz = pospres1.z - pospres2.z;

    	Real rr2 = drx*drx + dry*dry + drz*drz;

    	if(rr2 < sim_params.four_h_squared && rr2 > 1e-18)
    	{
    		//Add to particle neighbor count
    		neigh1 += j_index;
    		const Real prrhop1 = pospres1.w/(velrhop1.w * velrhop1.w);
    		const Real prrhop2 = pospres2.w/(velrhop2.w * velrhop2.w);
    		Real prs = -(prrhop1 + prrhop2);
    		const Real dvx = velrhop1.x - velrhop2.x, dvy =  velrhop1.y - velrhop2.y, dvz =  velrhop1.z - velrhop2.z;
    		Real wab,fac;
    		Real robar = ( velrhop1.w + velrhop2.w ) * 0.5;
			Real pi_visc = 0.0;
			const Real dot=drx*dvx + dry*dvy + drz*dvz;

    		{//====== Kernel =====
    			const Real rad=sqrt(rr2);
    			const Real qq=rad * sim_params.over_smoothing_length;
    			const Real wqq = 2.0 * qq + 1.0;
    			const Real wqq1 = 1.0 - 0.5 * qq;
    			const Real wqq2 = wqq1 * wqq1;
    			wab = sim_params.wendland_a1 * wqq * wqq2 * wqq2;
    			fac = sim_params.wendland_a2 * qq * wqq2 * wqq1/rad;

    		}


    	    //===== DeltaSPH =====
    //	    if(tdelta==DELTA_DBC || tdelta==DELTA_DBCExt)
    //	    {
    //			const Real rhop1over2=rhopp1/velrhop2.w;
    //			const Real visc_densi=CTE.delta2h*cbar*(rhop1over2-1)/(rr2+CTE.eta2);
    //			const Real dot3=(drx*frx+dry*fry+drz*frz);
    //			const Real delta=visc_densi*dot3*massp2;
    //			deltap1=(bound? FLT_MAX: deltap1+delta);
    //	    }



    	    {//-Artificial viscosity.

    			if( dot < 0.0 )
    			{

    	    	    const Real csoun1 = velrhop1.w * sim_params.over_rhop0;
					const Real csoun2 = velrhop2.w * sim_params.over_rhop0;
					const Real cbar= sim_params.cs0 * ((csoun1 * csoun1 * csoun1)+ (csoun2 * csoun2 * csoun2) ) * 0.5;

					const Real mu = sim_params.smoothing_length * dot/(rr2 + sim_params.epsilon);

					pi_visc = (-sim_params.nu * cbar * mu / robar);

    			}

    			//visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt); // <----- Reduction to only one value. Used for adaptive time stepping.
    	     }

    	    //===== XSPH correction =====
			  if(exec_params.xsph){
				const Real wab_rhobar = massp2*(wab/robar);
				vcor.x-=wab_rhobar * dvx;
				vcor.y-=wab_rhobar * dvy;
				vcor.z-=wab_rhobar * dvz;
			  }


			{// Density Derivative

				arp1 += massp2 * fac * dot;
			}

    		{// Acceleration

    			acep1.x += massp2 * (prs - pi_visc) * fac * drx;
    			acep1.y += massp2 * (prs - pi_visc) * fac * dry;
    			acep1.z += massp2 * (prs - pi_visc) * fac * drz;

    		}

    	}
    }

    __device__
    void interact_with_cell(uint gridHash, //
    						uint index, // index of particle i
    						Real  massp1, // mass of particle i
    						uint   *type, // Ordered particle type data for all particles
    						Real4 pospres1, // position vector and pressure of particle i
    						Real4 velrhop1, // velocity and density of particle i
    						Real4 *pospres, // Ordered position and pressure data for all particles
    						Real4 *velrhop, // Ordered velocity and density data for all particles
    						Real4 *ace_drhodt, //
    						Real3 &acep1, // Acceleration accumulator for particle i
    						Real &arp1, // Density derivative accumulator for particle i
    						Real  &visc, // Max dt for particle i based on viscous considerations
    						uint *cellStart, // Index of 1st particle in each grid cell
    						uint *cellEnd,
    						uint &neigh1,
    						Real3 &vcor,
    						uint *gridParticleIndex) // Index of last particle in each grid cell
    {


//    	uint gridHash = calcGridHash(gridPos);

    	Real massp2;

    	// get start of bucket for this cell
    	uint startIndex = cellStart[gridHash];

    	if (startIndex != 0xffffffff)          // cell is not empty

    	{

    		// iterate over particles in this cell
    		uint endIndex = cellEnd[gridHash];

    		for (uint j=startIndex; j<endIndex; j++)
    		{
    			if (j != index)                // check not interacting with self
    			{

    				uint j_index = gridParticleIndex[j];
    				Real4 pospres2 = pospres[j];
    				Real4 velrhop2 = velrhop[j];

    				uint type2 = type[j];

    				if(type2 == FLUID)
    				{
    					massp2 = sim_params.fluid_mass;
    				}
    				else
    				{
    					massp2 = sim_params.boundary_mass;
    				}

    				massp2 = sim_params.fluid_mass;

    				// collide two particles
    				particle_particle_interaction(pospres1, velrhop1, massp1,
    											  pospres2, velrhop2, massp2,
    											  acep1, arp1, visc,neigh1,vcor,j_index);


    			}
    		}

    	}

    }

    __global__
    void compute_particle_interactions(Real4 *ace_drhodt, // output: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
    								   Real4 *velrhop, // input: sorted velocity and density (v.x,v.y,v.z,rhop)
    								   Real4 *pospres, // input: sorted particle positions and pressures
    								   Real4 *velxcor, // output: velocity correction for xsph variant
    								   uint *gridParticleIndex, // input: sorted particle indicies
    								   uint *cellStart,
    								   uint *cellEnd,
    								   uint  *type, // input: sorted particle type (e.g. fluid, boundary, etc.)
    								   Real *viscdt, // output: max time step for adaptive time stepping
    								   uint numParticles,
    								   uint *neighbors,
    								   uint *gridParticleHash)

    {
        uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

        if (index < numParticles)
        {

        // read particle data from sorted arrays
        Real4 pospres1 =pospres[index];
        Real4 velrhop1 = velrhop[index];

        Real massp1,massp2;
        uint hash1 = gridParticleHash[index];

        if (type[index] == FLUID)
        {
        	massp1 = sim_params.fluid_mass;

        }
        else
        {
        	massp1 = sim_params.boundary_mass;
        }

        Real  visc = 0.0; // Holds max dt value based on viscous considerations

        // get address in grid
        int3 gridPos;
        gridPos.x = hash1%domain_params.grid_size.x;
        gridPos.z = int(hash1/(domain_params.grid_size.x*domain_params.grid_size.y));
        gridPos.y = int((hash1%(domain_params.grid_size.x*domain_params.grid_size.y))/domain_params.grid_size.x);

        int cxini = gridPos.x - min(gridPos.x,1);
        int cxfin = gridPos.x + min(domain_params.grid_size.x - gridPos.x - 1,1) + 1;
        int yini = gridPos.y - min(gridPos.y,1);
        int yfin = gridPos.y + min(domain_params.grid_size.y - gridPos.y - 1,1) + 1;
        int zini = gridPos.z - min(gridPos.z,1);
        int zfin = gridPos.z + min(domain_params.grid_size.y - gridPos.z - 1,1) + 1;

        // examine neighboring cells
        Real3 acep1 = make_Real3(sim_params.gravity.x,sim_params.gravity.y,sim_params.gravity.z); // Acceleration accumulator for particle(index)
        Real3 velcorr1 = make_Real3(0.0, 0.0, 0.0); // Velocity correction accumulator for particle(index)
        Real  arp1 = 0.0; // drho_dt accumulator for particle(index)
        uint  neigh1 = 0; // neighbor accumulator

        for (int z=zini; z<zfin; z++)
        {
            for (int y=yini; y<yfin; y++)
            {
                for (int x=cxini; x<cxfin; x++)
                {
                    int3 neighbourPos = make_int3(x, y, z);

                    uint hash2 = calcGridHash(neighbourPos);

                    	interact_with_cell(hash2,
                    					   index,
                    					   massp1,
                    					   type,
                    					   pospres1,
                    					   velrhop1,
                    					   pospres,
                    					   velrhop,
                    					   ace_drhodt,
                    					   acep1,
                    					   arp1,
                    					   visc,
                    					   cellStart,
                    					   cellEnd,
                    					   neigh1,
                    					   velcorr1,
                    					   gridParticleIndex);


                }
            }
        }

        // write accelerations back to original unsorted location
        uint originalIndex = gridParticleIndex[index];
        ace_drhodt[originalIndex] = make_Real4(acep1,arp1);
//        velxcor[originalIndex] = make_Real4(0.0, 0.0, 0.0,0.0);
        velxcor[originalIndex] = make_Real4(velcorr1,0.0);
        viscdt[originalIndex] = visc;
        neighbors[originalIndex] = neigh1;
    }
    }

    __global__
    void pre_interactionD(Real4 *ace_drhodt, // output: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
    				   Real4 *velrhop, // input: sorted velocity and density (v.x,v.y,v.z,rhop)
    				   Real4 *pospres, // input: sorted particle positions and pressures
    				   Real *viscdt, // output: max time step for adaptive time stepping
    				   uint numParticles)

    {
        uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

        if (index >= numParticles) return;

//        ace_drhodt[index].x = sim_params.gravity.x; ace_drhodt[index].y = sim_params.gravity.y; ace_drhodt[index].z = sim_params.gravity.z; ace_drhodt[index].w = 0.0; // initialize acceleration accumulators to zero
        viscdt[index] = 0.0; // initialize viscous time step tracker to zero

    }

    __global__
    void reduceAccelD(Real4 *ace_drhodt, // output: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
    				  Real4 *velrhop, // input: particle velocity and density
    				   Real *forcedt, // output: max time step for adaptive time stepping
    				   Real *viscdt,
    				   uint numParticles)

    {
        uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

        if (index >= numParticles) return;
        // Dt based on particle acceleration
        Real fa = ace_drhodt[index].x*ace_drhodt[index].x + ace_drhodt[index].y*ace_drhodt[index].y + ace_drhodt[index].z*ace_drhodt[index].z;
        Real mag_fa = sqrt(fa);
        Real dt = sqrt(sim_params.smoothing_length/mag_fa);
        forcedt[index] = dt;

        // Dt based on viscous diffusion
        const Real csound = sim_params.cs0*velrhop[index].w * sim_params.over_rhop0;
        dt = sim_params.smoothing_length/(csound + viscdt[index]);
        viscdt[index] = dt;
    }

    __global__
    void zero_accelerationD(Real4 *ace_drhodt)
    {
    	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    	if (index >= sim_params.num_boundary_particles) return;

    	// set acceleration of static boundary particles to zero.
    	ace_drhodt[index].x = 0.0; ace_drhodt[index].y = 0.0; ace_drhodt[index].z = 0.0;
    }

    __global__
    void zero_ycomponentD(Real4 *data, uint numParticles)
    {
    	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    	if (index >= numParticles) return;

    	// set acceleration of y-component to zero.
    	data[index].y = 0.0;
    }

    __device__
    int cellExists(int3 gridPos)
    {
    	// Checks grid position against grid limits
    	if( (gridPos.x >= 0) && (gridPos.x < domain_params.grid_size.x) && (gridPos.y >= 0) && (gridPos.y < domain_params.grid_size.y)
    			&& (gridPos.z >= 0) && (gridPos.z < domain_params.grid_size.z))
    	{
    		return 1;
    	}
    	else
    	{
    		return 0;
    	}

    }

    __global__
     void sheppard_density_filterD(Real4 *velrhop, // output: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
    		 	 	 	 	 	   	   Real4 *pospres, // output: new pressures
     								   Real4 *sorted_velrhop, // input: sorted velocity and density (v.x,v.y,v.z,rhop)
     								   Real4 *sorted_pospres, // input: sorted particle positions and pressures
     								   uint *gridParticleIndex, // input: sorted particle indicies
     								   uint *cellStart,
     								   uint *cellEnd,
     								   uint  *type, // input: sorted particle type (e.g. fluid, boundary, etc.)
     								   uint numParticles)

     {
         uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

         if (index < numParticles)
         {

         // read particle data from sorted arrays
         Real4 pospres1 = FETCH(sorted_pospres,index);
         Real4 velrhop1 = FETCH(sorted_velrhop,index);
         Real rhop1 = velrhop1.w;

         // get address in grid
         Real3 pos = make_Real3(pospres1.x,pospres1.y,pospres1.z);
         int3 gridPos = calcGridPos(pos);

         // examine neighboring cells
         Real sum_W = 0.0; // Acceleration accumulator for sum(W_ij)
         Real  sum_mass_W = 0.0; // Accumulator for sum(mass*W_ij)

         for (int z=-1; z<=1; z++)
         {
             for (int y=-1; y<=1; y++)
             {
                 for (int x=-1; x<=1; x++)
                 {
                     int3 neighbourPos = gridPos + make_int3(x, y, z);

                     // Check to see if cell exists
                     if(cellExists(neighbourPos))
                     {

                     	sheppard_interact_with_cell(neighbourPos,
                     					   index,
                     					   type,
                     					   pospres1,
                     					   velrhop1,
                     					   sorted_pospres,
                     					   sorted_velrhop,
                     					   sum_W,
                     					   sum_mass_W,
                     					   cellStart,
                     					   cellEnd);

                     }
                 }
             }
         }

		// write accelerations back to original unsorted location
		uint originalIndex = gridParticleIndex[index];
		uint type1 = FETCH(type,index);
		Real massp1;
		if(type1 == FLUID)
		{
			massp1 = sim_params.fluid_mass;
			Real rvol = massp1/rhop1;
			// Compute new density, along with contribution from the particle itself
			rhop1 = ((sum_W + sim_params.sheppard)*massp1)/(sum_mass_W + sim_params.sheppard*rvol);
			velrhop[originalIndex].w = rhop1;
			pospres[originalIndex].w = sim_params.b_coeff * ( pow(rhop1 * sim_params.over_rhop0,sim_params.gamma) - 1.0 ); // Compute particle pressure using Tait EOS
		}
     }
     }

    __device__
    void sheppard_interact_with_cell(int3 gridPos, //
    						uint index, // index of particle i
    						uint   *type, // Ordered particle type data for all particles
    						Real4 pospres1, // position vector and pressure of particle i
    						Real4 velrhop1, // velocity and density of particle i
    						Real4 *pospres, // Ordered position and pressure data for all particles
    						Real4 *velrhop, // Ordered velocity and density data for all particles
    						Real &sum_W, // Acceleration accumulator for particle i
    						Real &sum_mass_W, // Density derivative accumulator for particle i
    						uint *cellStart, // Index of 1st particle in each grid cell
    						uint *cellEnd) // Index of last particle in each grid cell
    {


    	uint gridHash = calcGridHash(gridPos);
    	Real massp2;

    	// get start of bucket for this cell
    	uint startIndex = FETCH(cellStart, gridHash);

    	if (startIndex != 0xffffffff)          // cell is not empty
    	{

    		// iterate over particles in this cell
    		uint endIndex = FETCH(cellEnd, gridHash);

    		for (uint j=startIndex; j<endIndex; j++)
    		{
    			if (j != index)                // check not interacting with self
    			{

    				Real4 pospres2 = FETCH(pospres,j);
    				Real4 velrhop2 = FETCH(velrhop,j);

    				uint type2 = FETCH(type,j);

    				if(type2 == FLUID)
    				{
    					massp2 = sim_params.fluid_mass;
    				}
    				else
    				{
    					massp2 = sim_params.boundary_mass;
    				}

    				massp2 = sim_params.fluid_mass;

    				// collide two particles
    			   	Real drx = pospres1.x - pospres2.x;
					Real dry = pospres1.y - pospres2.y;
					Real drz = pospres1.z - pospres2.z;

					Real rr2 = drx*drx + dry*dry + drz*drz;

					if(rr2<=sim_params.four_h_squared && rr2 >=1e-18)
					{
			    		Real wab;

			    		{//====== Kernel =====
			    			const Real rad=sqrt(rr2);
			    			const Real qq=rad * sim_params.over_smoothing_length;

			    			const Real wqq = 2.0 * qq + 1.0;
			    			const Real wqq1 = 1.0 - 0.5 * qq;
			    			const Real wqq2 = wqq1 * wqq1;
			    			wab = sim_params.wendland_a1 * wqq * wqq2 * wqq2;

			    			sum_W += wab;
			    			sum_mass_W += wab*(massp2/velrhop2.w);
			    		}
					}


    			}
    		}

    	}

    }

    __global__
    void predictorStepD(Real4 *pospres, // input: particle positions and pressures before predictor step
			   Real4 *velrhop, // input: particle velocity and density before predictor step
			   Real4 *pospres_Pre, // output: particle positions and pressures at half time step
			   Real4 *velrhop_Pre, // output: velocity and density (v.x,v.y,v.z,rhop) at half time step
			   Real4 *velxcor, // input: velocity correction for xsph variant
			   Real4 *ace_drhodt, // input: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
			   uint numParticles)
    {
    	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    	Real dt = 0.5 * exec_params.fixed_dt;
    	if (index < numParticles)
    	{

    		if (index < sim_params.num_boundary_particles) // Integrate boundary particle densities forward a half time step
    		{
    			Real rhop = velrhop[index].w;
    	        rhop += ace_drhodt[index].w * dt;
    	        velrhop_Pre[index].w = (rhop < sim_params.rhop0? sim_params.rhop0: rhop);     //-To prevent absorption of fluid particles by boundaries.
    	        pospres_Pre[index] = pospres[index];
    	        velrhop_Pre[index].x = 0.0; velrhop_Pre[index].y = 0.0; velrhop_Pre[index].z = 0.0;
    	    }
    	    else // Integrate Fluid particles forward a half time step
    	    {
    	    	Real3 dvdt = make_Real3(ace_drhodt[index].x,ace_drhodt[index].y,ace_drhodt[index].z);
    	   		Real3 pos = make_Real3(pospres[index].x, pospres[index].y, pospres[index].z);
				Real3 vel = make_Real3(velrhop[index].x, velrhop[index].y, velrhop[index].z);
				Real3 velcorr = make_Real3(velxcor[index].x, velxcor[index].y, velxcor[index].z);
    	    	Real rhop = velrhop[index].w;

				rhop += ace_drhodt[index].w * dt;
				pos += (vel + velcorr * sim_params.eps) * dt;
				vel += dvdt * dt;

				pospres_Pre[index].x = pos.x; pospres_Pre[index].y = pos.y; pospres_Pre[index].z = pos.z;
				velrhop_Pre[index].x = vel.x; velrhop_Pre[index].y = vel.y; velrhop_Pre[index].z = vel.z;
				velrhop_Pre[index].w = rhop;

//
    	    }

            pospres_Pre[index].w = sim_params.b_coeff * ( pow(velrhop_Pre[index].w * sim_params.over_rhop0,sim_params.gamma) - 1.0 ); // Compute particle pressure using Tait EOS


		}

    }

    __global__
     void correctorStepD(Real4 *pospres, // output: particle positions and pressures after predictor step
			   Real4 *velrhop, // output: particle velocity and density after predictor step
 			   Real4 *pospres_Pre, // input: particle positions and pressures
 			   Real4 *velrhop_Pre, // input: velocity and density (v.x,v.y,v.z,rhop)
 			   Real4 *ace_drhodt, // input: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
 			   uint numParticles)
     {

     	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
     	Real dt = exec_params.fixed_dt;

     	if (index < numParticles)
     	{

     		if (index < sim_params.num_boundary_particles) // Integrate boundary particle densities forward a half time step
			{
     			Real rhop = velrhop[index].w + dt * ace_drhodt[index].w;
				velrhop[index].w = (rhop < sim_params.rhop0? sim_params.rhop0: rhop);     //-To prevent absorption of fluid particles by boundaries.
			}

     		else // Integrate Fluid particles forward a half time step
     	    {

				Real3 vel = make_Real3(velrhop[index].x,velrhop[index].y,velrhop[index].z);
				Real3 pos = make_Real3(pospres[index].x,pospres[index].y,pospres[index].z);
				Real3 dvdt = make_Real3(ace_drhodt[index].x,ace_drhodt[index].y,ace_drhodt[index].z);

				Real rhop = velrhop[index].w + dt * ace_drhodt[index].w;

				Real3 velpre = make_Real3(velrhop_Pre[index].x,velrhop_Pre[index].y,velrhop_Pre[index].z);
				vel += dvdt*dt;
				pos += velpre*dt;

				pospres[index].x = pos.x; pospres[index].y = pos.y; pospres[index].z = pos.z;
				velrhop[index].x = vel.x; velrhop[index].y = vel.y; velrhop[index].z = vel.z;
				velrhop[index].w = rhop;
     	    }

     		pospres[index].w = sim_params.b_coeff * ( pow(velrhop[index].w * sim_params.over_rhop0,sim_params.gamma) - 1.0 ); // Compute particle pressure using Tait EOS

 		}

     }

}   // namespace gpusph
