/**
 * CUDA header for running parallel cellular automaton.
 * @author Logan Apple
 * @date 5/15/2020
 */

#ifndef GOL_DEVICE_CUH
#define GOL_DEVICE_CUH

#include <cstdio>
#include <cstdlib>

#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/* Modified from https://bit.ly/365DwFs */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, 
                      int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", 
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void call_cuda_naive_gol_update(int num_threads,
                        int width, int height,
                        uint8_t* cells, uint8_t* updated_cells);

void call_cuda_opt_gol_update(int num_threads,
                        int width, int height,
                        uint8_t* init_cells,
                        int iterations, 
                        thrust::host_vector<uint8_t*> host_history);

#endif