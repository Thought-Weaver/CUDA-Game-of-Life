/**
 * Main CUDA file for running parallel cellular automaton.
 * @author Logan Apple
 * @date 5/15/2020
 */

#include "gol.cuh"

#include <cstdio>
#include <cuda_runtime.h>

__global__ void naive_gol_update_kernel();

__global__ void cuda_optimized_gol_update_kernel();

void call_cuda_gol_update();