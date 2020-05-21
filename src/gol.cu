/**
 * Main CUDA file for running parallel cellular automaton.
 * @author Logan Apple
 * @date 5/15/2020
 */

#include "gol.cuh"

// Trying texture memory -- probably not better than just using shared memory,
// but it's worth a shot.
// Also, I feel like this shouldn't be global, but all the CUDA examples have
// it as such?
texture<uint8_t, 2, cudaReadModeElementType> texmem;

// What if I just passed the grid instead?
__host__ __device__ uint8_t count_neighbors(int x, int y, 
                                        int width, int height, 
                                        uint8_t* cells) {
    uint8_t alive = 0;
    
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
                                    uint8_t* cells, uint8_t* updated_cells) {
    const int num_threads_x = blockDim.x * gridDim.x;
    const int num_threads_y = blockDim.y * gridDim.y;

    // Thread indices.
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (; tidy < height; tidy += num_threads_y) {
        for (; tidx < width; tidx += num_threads_x) {
            uint8_t neighbors = count_neighbors(tidx, tidy, width, height, cells);
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
                                        uint8_t* updated_cells) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    uint8_t neighbors = 0;
            
    // Take advantage of loop unrolling to make this faster.
    #pragma unroll
    for (int x = -1; x <= 1; ++x) {
        #pragma unroll
        for (int y = -1; y <= 1; ++y) {
            int y2 = tidy + y;
            int x2 = tidx + x;
            if (y2 != y || x2 != x) {
                if (y2 >= 0 && y2 < height && 
                    x2 >= 0 && x2 < width) {
                    neighbors += tex2D(texmem, x2, y2);
                }
            }
        }
    }

    // Any live cell with two or three neighbors survives.
    if (tex2D(texmem, tidx, tidy) == 1 && 
            (neighbors == 2 || neighbors == 3)) {
        updated_cells[tidy * width + tidx] = 1;
    }
    // Any dead cell with three live neighbors comes to life.
    else if (tex2D(texmem, tidx, tidy) == 0 && neighbors == 3) {
        updated_cells[tidy * width + tidx] = 1;
    }
    // Any other cells die.
    else {
        updated_cells[tidy * width + tidx] = 0;
    }
}

void call_cuda_gol_update(int num_threads,
                          int width, int height,
                          uint8_t* cells, uint8_t* updated_cells,
                          bool optimized) {
    // Maybe I should fix these rather than let the user specify them?
    dim3 block_size(num_threads, num_threads);
    dim3 grid_size(int((width + num_threads - 1) / num_threads), 
                   int((height + num_threads - 1) / num_threads));
    if (optimized) {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uint8_t>();

        size_t pitch;
        cudaMallocPitch(&cells, &pitch, sizeof(uint8_t) * width, height);
        cudaBindTexture2D(0, texmem, cells, desc, width, height, pitch);

        optimized_update_kernel<<<grid_size, block_size>>>(width, height, 
            updated_cells);
    }
    else {
        naive_update_kernel<<<grid_size, block_size>>>(width, height, 
            cells, updated_cells);
    }
}