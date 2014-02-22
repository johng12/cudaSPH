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

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayFromDevice(void *host, const void *device, int offset, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);

    void setParameters(SimParams *hostParams);

    void integrateSystem(double *pos,
                         double *vel,
                         double deltaTime,
                         uint numParticles);

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  double *pos,
                  int    numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     double *sortedPos,
                                     double *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     double *oldPos,
                                     double *oldVel,
                                     uint   numParticles,
                                     uint   numCells);

    void collide(double *newVel,
                 double *sortedPos,
                 double *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells);

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

}
