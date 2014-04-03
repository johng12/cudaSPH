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

#include <helper_functions.h>
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"

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
    //    template < typename T >
    //    T cuda_sum(T *data, uint numElements)
    //    {
    //    	T result = thrust::reduce(thrust::device_ptr<T>(data),thrust::device_ptr<T>(data + numElements));
    //    	return result;
    //    }
    //
    //    template < typename T >
    //    T cuda_max(T *data, uint numElements)
    //    {
    //    	T result = thrust::max_element(thrust::device_ptr<T>(data),thrust::device_ptr<T>(data + numElements));
    //    	return result;
    //    }
    //
    //    template < typename T >
    //    T cuda_min(T *data, uint numElements)
    //    {
    //    	T result = thrust::min_element(thrust::device_ptr<T>(data),thrust::device_ptr<T>(data + numElements));
    //		return result;
    //    }
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

//    template < typename T >
//    T cuda_sum(T *data, uint numElements)
//    {
//    	T result = thrust::reduce(thrust::device_ptr<T>(data),thrust::device_ptr<T>(data + numElements));
//    	return result;
//    }
//
//    double
//    cuda_max(double *data, uint numElements)
//    {
//
//    	thrust::host_vector<double> h_result;
//    	thrust::device_ptr<double> result = thrust::max_element(thrust::device_ptr<double>(data),thrust::device_ptr<double>(data + numElements));
//    	h_result = result;
//    	return h_result;
//
//
//    }

//
//    template < typename T >
//    T cuda_min(T *data, uint numElements)
//    {
//    	T result = thrust::min_element(thrust::device_ptr<T>(data),thrust::device_ptr<T>(data + numElements));
//		return result;
//    }

}   // namespace gpusph
