/**
 * CUDA header for running parallel cellular automaton.
 * @author Logan Apple
 * @date 5/15/2020
 */

#ifndef GOL_DEVICE_CUH
#define GOL_DEVICE_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void call_cuda_gol_update(int blocks,
                          int width, int height,
                          int* cells, int* updated_cells,
                          bool optimized);

#endif