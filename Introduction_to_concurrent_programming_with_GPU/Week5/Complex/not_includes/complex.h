#include <stdio.h>
#include <tuple>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__global__ void vectorMult(const float *A, const float *B, float *C, int numElements);
float deviceMultiply(float a, float b);
std::tuple<float *, float *, float *> allocateHostMemory(int numElements);
std::tuple<float *, float *, float *> allocateDeviceMemory(int numElements);
void copyFromHostToDevice(float *h_A, float *h_B, float *d_A, float *d_B, int numElements);
void executeKernel(float *d_A, float *d_B, float *d_C);
__host__ void copyFromDeviceToHost(float *d_C, float *h_C, int numElements);
void deallocateMemory(float *h_A, float *h_B, float *h_C, float *d_A, float *d_B, float *d_C);
void cleanUpDevice();
void performTest(float *h_A, float *h_B, float *h_C, int numElements);