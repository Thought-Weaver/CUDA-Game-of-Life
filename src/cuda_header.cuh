/**
 * CUDA header for making function declarations nicer.
 * @author Logan Apple
 * @date 5/15/2020
 */

#ifndef CUDA_HEADER_CUH
#define CUDA_HEADER_CUH

#ifdef __CUDA_ARCH__

// Device function attributes.
#include <cuda_runtime.h>
#define CUDA_CALLABLE __host__ __device__

#else

// Host function attributes.
#define CUDA_CALLABLE

#endif // __CUDA_ARCH__

#endif // CUDA_HEADER_CUH
