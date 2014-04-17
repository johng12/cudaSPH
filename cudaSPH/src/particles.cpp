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

#define NUM_PARTICLES   16000

uint numParticles = 0;
int numIterations = 0; // run until exit

// simulation parameters
Real timestep = 0.5;
Real damping = 1.0;
Real gravity = 0.0003;
int iterations = 1;
int ballr = 10;

Real collideSpring = 0.5;
Real collideDamping = 0.02;
Real collideShear = 0.1;
Real collideAttraction = 0.0;

ParticleSystem *psystem = 0;
StopWatchInterface *timer = NULL;

// Auto-Verification Code
unsigned int g_TotalErrors = 0;
char        *g_refFile = NULL;

const char *sSDKsample = "CUDA Particles Simulation";

// initialize particle system
void initParticleSystem(int numParticles)
{
    psystem = new ParticleSystem(numParticles);
    psystem->load("Fluid.asc");
    psystem->dumpParticles(0,psystem->getNumParticles(),"PART_0000.dat");
    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
}

void runBenchmark(int iterations, char *exec_path)
{
    printf("Run %u particles simulation for %d iterations...\n\n", numParticles, iterations);
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);
    int printStep = 100;

    char buffer[32]; // The filename buffer.

    psystem->dumpParticles(0,psystem->getNumParticles(),"PART_0000.dat");

    for (int i = 0; i < iterations; ++i)
    {
        psystem->update(timestep);

        if(!(i%printStep)){
        	snprintf(buffer, sizeof(char) * 32, "PART%i.dat", i);
        	psystem->dumpParticles(0,psystem->getNumParticles(),buffer);
        }

    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    Real fAvgSeconds = ((Real)1.0e-3 * (Real)sdkGetTimerValue(&timer)/(Real)iterations);
    psystem->dumpParameters();
    printf("particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-3 * numParticles)/fAvgSeconds, fAvgSeconds, numParticles, 1, 0);
//    if (g_refFile)
//    {
//        printf("\nChecking result...\n\n");

//        Real *hPos = (Real *)malloc(sizeof(Real)*4*psystem->getNumParticles());
//        hPos = psystem->getArray(ParticleSystem::POSITION);
//                            0, sizeof(Real)*4*psystem->getNumParticles());

//        Real *hPos = (Real *)malloc(sizeof(Real)*4*psystem->getNumParticles());
//        copyArrayFromDevice(hPos, psystem->getCudaPosVBO(),
//                            0, sizeof(Real)*4*psystem->getNumParticles());

//

//        sdkDumpBin((void *)hPos, sizeof(Real)*4*psystem->getNumParticles(), "particles.bin");

//        sdkDumpBin((void *)hPos, sizeof(Real)*4*psystem->getNumParticles(), "particles.bin");

//
//        if (!sdkCompareBin2BinFloat("particles.bin", g_refFile, sizeof(float)*4*psystem->getNumParticles(),
//                                    MAX_EPSILON_ERROR, THRESHOLD, exec_path))
//        {
//            g_TotalErrors++;
//        }
//    }
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
    printf("%s Starting...\n\n", sSDKsample);

    numParticles = NUM_PARTICLES;
    numIterations = 0;

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "n"))
        {
            numParticles = getCmdLineArgumentInt(argc, (const char **)argv, "n");
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "file", &g_refFile);
            numIterations = 1;
        }
    }

    printf("particles: %d\n", numParticles);

    bool benchmark = checkCmdLineFlag(argc, (const char **) argv, "benchmark") != 0;

    if (checkCmdLineFlag(argc, (const char **) argv, "i"))
    {
        numIterations = getCmdLineArgumentInt(argc, (const char **) argv, "i");
    }

    if (g_refFile)
    {
        gpusph::cudaInit(argc, argv);
    }
    else
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            printf("[%s]\n", argv[0]);
            printf("   Does not explicitly support -device=n in OpenGL mode\n");
            printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
            printf(" > %s -device=n -file=<*.bin>\n", argv[0]);
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    initParticleSystem(numParticles);

    if (benchmark)
    {
        if (numIterations <= 0)
        {
            numIterations = 300;
        }

//        runBenchmark(numIterations, argv[0]);
    }

    if (psystem)
    {
        delete psystem;
    }

    cudaDeviceReset();
    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}

