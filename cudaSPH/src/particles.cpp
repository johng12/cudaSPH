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

/*
    Particle system example with collisions using uniform grid

    CUDA 2.1 SDK release 12/2008
    - removed atomic grid method, some optimization, added demo mode.

    CUDA 2.2 release 3/2009
    - replaced sort function with latest radix sort, now disables v-sync.
    - added support for automated testing and comparison to a reference value.
*/

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "particleSystem.h"

#define MAX_EPSILON_ERROR 5.00
#define THRESHOLD         0.30

#define NUM_PARTICLES   20161

uint numParticles = 0;
int numIterations = 1; // run until exit

// simulation parameters
Real timestep = 1.0e-5;
Real save_interval;
Real print_interval;
Real simulation_duration;

ParticleSystem *psystem = 0;
StopWatchInterface *timer = NULL;

// Auto-Verification Code
unsigned int g_TotalErrors = 0;
char        *cfgFileName = NULL;
char *meshFileName = NULL;

const char *programName = "GPUSPH Simulation";

// initialize particle system
void initParticleSystem(const char *cfgFileName, const char *meshFileName)
{

    psystem = new ParticleSystem(cfgFileName);
    psystem->load(meshFileName);

    numParticles = psystem ->getNumParticles();
    save_interval = psystem ->getSaveInterval();
    print_interval = psystem ->getPrintInterval();
    simulation_duration = psystem ->getSimulationDuration();
    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
}

void runCase(char *exec_path)
{
    printf("Run %u particles simulation for %f seconds...\n", numParticles, simulation_duration);
    printf("Save interval: %e seconds...\n",save_interval);
    printf("Printscreen interval: %e seconds...\n\n",print_interval);
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);
    int iteration = 1;
    int part = 1;
    int print = 1;
    Real current_time = 0.0;
    char buffer[32]; // The filename buffer.

    psystem ->dumpParameters();
    printf("*************************************************\n\n");
    printf("iteration       current time       time step \n");
    printf("---------------------------------------------\n");
    snprintf(buffer, sizeof(char) * 32, "PART_%04i.dat", 0);
	psystem->dumpParticles(0,psystem->getNumParticles(),current_time,buffer);

    while (current_time < simulation_duration)
    {
        psystem->update(timestep);
        current_time = current_time + timestep;
        if(current_time - save_interval*(part - 1) >= save_interval )
        {
        	snprintf(buffer, sizeof(char) * 32, "PART_%04i.dat", part);
        	psystem->dumpParticles(0,psystem->getNumParticles(),current_time,buffer);
        	part++;
        }

        if(current_time - print_interval*(print - 1) >= print_interval )
        {
        	if(!(print%10))
        	{
				printf("\n iteration       current time       time step \n");
				printf("---------------------------------------------\n");
        	}
        	printf(" %d              %5.4e              %5.4e \n",iteration,current_time,timestep);
        	print++;
        }

//        if(!(iteration%filterStep)){
//        	psystem->apply_sheppard_filter();
//        }

        iteration++;
    }

    snprintf(buffer, sizeof(char) * 32, "PART_%04i.dat", part);
    psystem->dumpParticles(0,psystem->getNumParticles(),current_time,buffer);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    Real fAvgSeconds = ((Real)1.0e-3 * (Real)sdkGetTimerValue(&timer)/(Real)iteration);
    psystem ->dumpParameters();
    printf("particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-3 * numParticles)/fAvgSeconds, fAvgSeconds, numParticles, 1, 0);

}

inline Real frand()
{
    return rand() / (Real) RAND_MAX;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    printf("\n************************************ \n");
    printf("%s Starting...\n\n", programName);

//    numIterations = 0;
//    char *iter;


	if (checkCmdLineFlag(argc, (const char **)argv, "cfgFile"))
	{
		getCmdLineArgumentString(argc, (const char **)argv, "cfgFile", &cfgFileName);
	}
	else
	{
		printf("Error: cfgFile missing. \n");
		exit(1);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "meshFile"))
	{
		getCmdLineArgumentString(argc, (const char **)argv, "meshFile", &meshFileName);
	}
	else
	{
		printf("Error: meshFile missing. \n");
		exit(1);
	}


//    if (checkCmdLineFlag(argc, (const char **) argv, "iter"))
//    {
//    	getCmdLineArgumentString(argc, (const char **)argv, "iter", &iter);
//    	numIterations = (uint)atoi(iter);
//    }

    initParticleSystem(cfgFileName,meshFileName);
//
//	if (numIterations <= 0)
//	{
//		numIterations = 1;
//	}

	runCase(argv[0]);


    if (psystem)
    {
        delete psystem;
    }

    cudaDeviceReset();
    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}

