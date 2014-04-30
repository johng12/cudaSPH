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
                         Real *ace_drhodt,
                         Real deltaTime,
                         uint numParticles)
    {
        thrust::device_ptr<Real4> d_pos4((Real4 *)pospres);
        thrust::device_ptr<Real4> d_vel4((Real4 *)velrhop);
        thrust::device_ptr<Real4> d_pospre4((Real4 *)pospres_pre);
        thrust::device_ptr<Real4> d_velpre4((Real4 *)velrhop_pre);
        thrust::device_ptr<Real4> d_ace_drhodt4((Real4 *) ace_drhodt);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4,d_pospre4,d_velpre4,d_ace_drhodt4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles,d_pospre4+numParticles,d_velpre4+numParticles, d_ace_drhodt4+numParticles)),
            integrate_predictor(deltaTime));
    }

    void correctorStep(Real *pos,
                         Real *vel,
                         Real *pospre,
                         Real *velpre,
                         Real *ace_drhodt,
                         Real deltaTime,
                         uint numParticles)
    {
        thrust::device_ptr<Real4> d_pos4((Real4 *)pos);
        thrust::device_ptr<Real4> d_vel4((Real4 *)vel);
        thrust::device_ptr<Real4> d_pospre4((Real4 *)pospre);
		thrust::device_ptr<Real4> d_velpre4((Real4 *)velpre);
		thrust::device_ptr<Real4> d_ace_drhodt4((Real4 *)ace_drhodt);

        thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4,d_pospre4,d_velpre4,d_ace_drhodt4)),
                thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles,d_pospre4+numParticles,d_velpre4+numParticles,d_ace_drhodt4+numParticles)),
                integrate_corrector(deltaTime));
    }

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  Real *pos,
                  int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (Real4 *) pos,
                                               numParticles);

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
                 Real *sorted_velrhop,
                 Real *sorted_pospres,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   *sorted_type,
                 Real  *viscdt,
                 uint   numParticles,
                 uint   numCells)
    {
#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, sorted_pospres, numParticles*sizeof(Real4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, sorted_velrhop, numParticles*sizeof(Real4)));
        checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
        checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        // execute the kernel
        compute_particle_interactions<<< numBlocks, numThreads >>>((Real4 *) ace_drhodt,
																   (Real4 *) sorted_velrhop,
																   (Real4 *) sorted_pospres,
																   gridParticleIndex,
																   cellStart,
																   cellEnd,
																   sorted_type,
																   viscdt,
																   numParticles);

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
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
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

// Begin CUDA kernel code:


    // calculate position in uniform grid
    __device__ int3 calcGridPos(Real3 p)
    {
        int3 gridPos;
        gridPos.x = floor((p.x - domain_params.world_origin.x) / domain_params.cell_size.x);
        gridPos.y = floor((p.y - domain_params.world_origin.y) / domain_params.cell_size.y);
        gridPos.z = floor((p.z - domain_params.world_origin.z) / domain_params.cell_size.z);
        return gridPos;
    }

    // calculate address in grid from position (clamping to edges)
    __device__ uint calcGridHash(int3 gridPos)
    {
    //    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    //    gridPos.y = gridPos.y & (params.gridSize.y-1);
    //    gridPos.z = gridPos.z & (params.gridSize.z-1);
        return __umul24(__umul24(gridPos.z, domain_params.grid_size.y), domain_params.grid_size.x) + __umul24(gridPos.y, domain_params.grid_size.x) + gridPos.x;
    }

    // calculate grid hash value for each particle
    __global__
    void calcHashD(uint   *gridParticleHash,  // output
                   uint   *gridParticleIndex, // output
                   Real4 *pos,               // input: positions
                   uint    numParticles)
    {
        uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

        if (index >= numParticles) return;

        volatile Real4 p = pos[index];

        // get address in grid
        int3 gridPos = calcGridPos(make_Real3(p.x, p.y, p.z));
        uint hash = calcGridHash(gridPos);

        // store grid hash and particle index
        gridParticleHash[index] = hash;
        gridParticleIndex[index] = index;
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

            if (index > 0 && threadIdx.x == 0)
            {
                // first thread in block must load neighbor particle hash
                sharedHash[0] = gridParticleHash[index-1];
            }
        }

        __syncthreads();

        if (index < numParticles)
        {
            // If this particle has a different cell index to the previous
            // particle then it must be the first particle in the cell,
            // so store the index of this particle in the cell.
            // As it isn't the first particle, it must also be the cell end of
            // the previous particle's cell

            if (index == 0 || hash != sharedHash[threadIdx.x])
            {
                cellStart[hash] = index;

                if (index > 0)
                    cellEnd[sharedHash[threadIdx.x]] = index;
            }

            if (index == numParticles - 1)
            {
                cellEnd[hash] = index + 1;
            }

            // Now use the sorted index to reorder the pos and vel data
            uint sortedIndex = gridParticleIndex[index];
            Real4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
            Real4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh
            uint p_type = FETCH(oldType, sortedIndex);

            sortedPos[index] = pos;
            sortedVel[index] = vel;
            sortedType[index] = p_type;
        }


    }

    __device__
    void particle_particle_interaction(Real4 pospres1, Real4 velrhop1, Real massp1,
    								   Real4 pospres2, Real4 velrhop2, Real massp2,
    								   Real3 acep1, Real arp1, Real visc)
    {
    	Real drx = pospres1.x - pospres2.x;
    	Real dry = pospres1.y - pospres2.y;
    	Real drz = pospres1.z - pospres2.z;
    	const Real dvx = velrhop1.x - velrhop2.x, dvy =  velrhop1.y - velrhop2.y, dvz =  velrhop1.z - velrhop2.z;
    	Real rr2 = drx*drx + dry*dry + drz*drz;

    	if(rr2<=sim_params.four_h_squared && rr2 >=1e-18)
    	{
    		const Real prrhop2 = pospres2.w/(velrhop2.w * velrhop2.w);
    		const Real prrhop1 = pospres1.w/(velrhop1.w * velrhop1.w);
    		Real prs = prrhop1 + prrhop2;

    		Real wab,frx,fry,frz;

    		{//====== Kernel =====
    			const Real rad=sqrt(rr2);
    			const Real qq=rad * sim_params.over_smoothing_length;
    			Real fac;

    			const Real wqq = 2.0 * qq + 1.0;
    			const Real wqq1 = 1.0 - 0.5 * qq;
    			const Real wqq2 = wqq1 * wqq1;
    			wab = sim_params.wendland_a1 * wqq * wqq2 * wqq2;
    			fac = sim_params.wendland_a2 * qq * wqq2 * wqq1 / rad;

    			frx = fac * drx; fry = fac * dry; frz = fac * drz;
    		}

    		{// Acceleration
    			const Real p_vpm = -prs * massp2;
    			acep1.x += p_vpm * frx; acep1.y += p_vpm * fry; acep1.z += p_vpm * frz;
    		}

    		{// Density Derivative

    			arp1 += massp2 * (dvx * frx + dvy * fry + dvz * frz);
    		}

    		const Real csoun1 = velrhop1.w * sim_params.over_rhop0;
    		const Real csoun2 = velrhop2.w * sim_params.over_rhop0;
    	    const Real cbar=(sim_params.cs0 * (csoun1 * csoun1 * csoun1)+ sim_params.cs0 *(csoun2 * csoun2 * csoun2) ) * 0.5;

    	    //===== DeltaSPH =====
    //	    if(tdelta==DELTA_DBC || tdelta==DELTA_DBCExt)
    //	    {
    //			const Real rhop1over2=rhopp1/velrhop2.w;
    //			const Real visc_densi=CTE.delta2h*cbar*(rhop1over2-1)/(rr2+CTE.eta2);
    //			const Real dot3=(drx*frx+dry*fry+drz*frz);
    //			const Real delta=visc_densi*dot3*massp2;
    //			deltap1=(bound? FLT_MAX: deltap1+delta);
    //	    }

    	    Real robar = ( velrhop1.w + velrhop2.w ) * 0.5;

    	    {//===== Viscosity =====
    			const Real dot=drx*dvx + dry*dvy + drz*dvz;
    			const Real dot_rr2=dot/(rr2 + sim_params.epsilon);
    			//-Artificial viscosity.
    	//		if(tvisco==VISCO_Artificial && dot<0)
    			if( dot < 0.0 )
    			{
    			  const Real amubar = sim_params.smoothing_length * dot_rr2;
    			  const Real pi_visc = (-sim_params.nu * cbar * amubar / robar) * massp2;
    			  acep1.x-= pi_visc * frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
    			}

    			visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt); // <----- Reduction to only one value. Used for adaptive time stepping.
    	     }

    	}
    }

    __device__
    void interact_with_cell(int3 gridPos, //
    						uint index, // index of particle i
    						Real  massp1, // mass of particle i
    						uint   *type, // Ordered particle type data for all particles
    						Real4 pospres1, // position vector and pressure of particle i
    						Real4 velrhop1, // velocity and density of particle i
    						Real4 *pospres, // Ordered position and pressure data for all particles
    						Real4 *velrhop, // Ordered velocity and density data for all particles
    						Real4 *ace_drhodt, // Need to fix this term
    						Real3 acep1, // Acceleration accumulator for particle i
    						Real arp1, // Density derivative accumulator for particle i
    						Real  visc, // Max dt for particle i based on viscous considerations
    						uint *cellStart, // Index of 1st particle in each grid cell
    						uint *cellEnd) // Index of last particle in each grid cell
    {
    	uint gridHash = calcGridHash(gridPos);

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
    				Real massp2;
    				uint type2 = FETCH(type,j);

    				if(type2 == FLUID)
    				{
    					massp2 = sim_params.fluid_mass;
    				}
    				else
    				{
    					massp2 = sim_params.boundary_mass;
    				}

    				// collide two particles
    				particle_particle_interaction(pospres1, velrhop1, massp1,
    											  pospres2, velrhop2, massp2,
    											  acep1, arp1, visc);

    			}
    		}
    	}

    }

    __global__
    void compute_particle_interactions(Real4 *ace_drhodt, // output: acceleration and drho_dt values (a.x,a.y,a.z,drho_dt)
    								   Real4 *velrhop, // input: sorted velocity and density (v.x,v.y,v.z,rhop)
    								   Real4 *pospres, // input: sorted particle positions and pressures
    								   uint *gridParticleIndex, // input: sorted particle indicies
    								   uint *cellStart,
    								   uint *cellEnd,
    								   uint  *type, // input: sorted particle type (e.g. fluid, boundary, etc.)
    								   Real *viscdt, // output: max time step for adaptive time stepping
    								   uint numParticles)

    {
        uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

        if (index >= numParticles) return;

        // read particle data from sorted arrays
        Real4 pospres1 = FETCH(pospres,index);
        Real4 velrhop1 = FETCH(velrhop,index);
        Real4 ace_drhodt1 = FETCH(ace_drhodt,index); // *This is not sorted, need to work this out.
        Real massp1 = (FETCH(type,index)=FLUID? sim_params.fluid_mass: sim_params.boundary_mass);
        Real  visc = viscdt[index]; // Holds max dt value based on viscous considerations

        // get address in grid
        Real3 pos = make_Real3(pospres1.x,pospres1.y,pospres1.z);
        int3 gridPos = calcGridPos(pos);

        // examine neighboring cells
        Real3 acep1 = make_Real3(ace_drhodt1); // Acceleration accumulator for particle(index)
        Real  arp1 = 0.0; // drho_dt accumulator for particle(index)

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
                    	interact_with_cell(gridPos,
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
                    					   cellEnd);
                    }
                }
            }
        }

        // write new velocity back to original unsorted location
        uint originalIndex = gridParticleIndex[index];
        ace_drhodt[originalIndex] = make_Real4(acep1,arp1);
        viscdt[index] = visc;
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

        ace_drhodt[index].x = sim_params.gravity.x; ace_drhodt[index].y = sim_params.gravity.y; ace_drhodt[index].z = sim_params.gravity.z; ace_drhodt[index].w = 0.0; // initialize acceleration accumulators to zero
        viscdt[index] = 0.0; // initialize viscous time step tracker to zero
        Real4 pospres1 = FETCH(pospres,index);
        Real4 velrhop1 = FETCH(velrhop,index);

        pospres1.w = sim_params.b_coeff * ( pow(velrhop1.w * sim_params.over_rhop0,sim_params.gamma) - 1.0 ); // Compute particle pressure using Tait EOS
        pospres[index] = pospres1;
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
    	if( (gridPos.x >= 0) && (gridPos.x <= domain_params.grid_size.x - 1) && (gridPos.y >= 0) && (gridPos.y <= domain_params.grid_size.y - 1)
    			&& (gridPos.z >= 0) && (gridPos.z <= domain_params.grid_size.z - 1))
    	{
    		return 1;
    	}
    	else
    	{
    		return 0;
    	}

    }

}   // namespace gpusph
