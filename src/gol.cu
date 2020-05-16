/**
 * Main CUDA file for running parallel cellular automaton.
 * @author Logan Apple
 * @date 5/15/2020
 */

#include "gol.cuh"

#include <cstdio>
#include <cuda_runtime.h>

// What if I just passed the grid instead?
__host__ __device__ int count_neighbors(int x, int y, 
                                        int width, int height, 
                                        int* cells) {
    int alive = 0;
    
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0;
    }

    for (int i = y - 1; i <= y + 1; ++i) {
        for (int j = x - 1; j <= x + 1; ++j) {
            if (i != y || j != x) {
                if (i >= 0 && i < height && j >= 0 && j < width) {
                    alive += cells[i * width + j];
                }
            }
        }
    }

    return alive;
}

__global__ void naive_update_kernel(int width, int height, 
                                    float* cells, float* updated_cells) {
    const uint num_threads_x = blockDim.x * gridDim.x;
    const uint num_threads_y = blockDim.y * gridDim.y;

    // Thread indices.
    uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint tidy = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tidy < height; tidy += num_threads_y) {
        for (; tidx < width; tidx += num_threads_x) {
            int neighbors = count_neighbors(tidx, tidy, width, height, cells);
            // Any live cell with two or three neighbors survives.
            if (cells[tidy * width + tidx] == 1 && 
                (neighbors == 2 || neighbors == 3)) {
                updated_cells[tidy * width + tidx] = 1;
            }
            // Any dead cell with three live neighbors comes to life.
            else if (cells[tidy * width + tidx] == 0 && neighbors == 3) {
                updated_cells[tidy * width + tidx] = 1;
            }
            // Any other cells die.
            else {
                updated_cells[tidy * width + tidx] = 0;
            }
        }
    }
}

__global__ void optimized_update_kernel(int width, int height, 
                                        float* cells, float* updated_cells) {

}

void call_cuda_gol_update(uint blocks, uint threads_per_block,
                          int width, int height,
                          float* cells, float* out_cells,
                          bool optimized) {
    if (optimized) {
        optimized_update_kernel<<<gridSize, blockSize>>>(width, height, 
                                                         cells, updated_cells);
    }
    else {
        naive_update_kernel<<<gridSize, blockSize>>>(width, height,
                                                     cells, updated_cells);
    }
}