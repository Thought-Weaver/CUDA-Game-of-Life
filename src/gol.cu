/**
 * Main CUDA file for running parallel cellular automaton.
 * @author Logan Apple
 * @date 5/15/2020
 */

#include "gol.cuh"

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
        uint8_t* cells, uint8_t* updated_cells) {
    extern __shared__ uint8_t shmem[];

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    
    int i = threadIdx.y;
    int j = threadIdx.x;

    if (tidx >= 0 && tidx < width && tidy >= 0 && tidy < height) {
        shmem[i * width + j] = cells[tidy * width + tidx];
    }

    __syncthreads();

    if (tidx >= 0 && tidx < width && tidy >= 0 && tidy < height) {
        uint8_t neighbors = 0;

        // Take advantage of loop unrolling to make this faster.
        #pragma unroll
        for (int x = -1; x <= 1; ++x) {
            #pragma unroll
            for (int y = -1; y <= 1; ++y) {
                int y2 = i + y;
                int x2 = j + x;
                if (x != 0 || y != 0) {
                    if (y2 >= 0 && y2 < height && 
                            x2 >= 0 && x2 < width) {
                        neighbors += shmem[y2 * width + x2];
                    }
                }
            }
        }

        // Any live cell with two or three neighbors survives.
        if ((neighbors == 2 || neighbors == 3) && shmem[i * width + j] == 1) {
            updated_cells[tidy * width + tidx] = 1;
        }
        // Any dead cell with three live neighbors comes to life.
        else if (neighbors == 3 && shmem[i * width + j] == 0) {
            updated_cells[tidy * width + tidx] = 1;
        }
        // Any other cells die.
        else {
            updated_cells[tidy * width + tidx] = 0;
        }
    }
}

__global__ void optimized_update_kernel_bitwise(int width, int height,
    uint8_t* cells, uint8_t* updated_cells) {
    extern __shared__ uint8_t shmem[];

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if (tidx >= 0 && tidx < width && tidy >= 0 && tidy < height) {
        shmem[i * width + j] = cells[tidy * width + tidx];
    }

    __syncthreads();

    if (tidx >= 0 && tidx < width && tidy >= 0 && tidy < height) {
        for (int k = 0; k < 8; ++k) {
            uint8_t current = (shmem[i * width + j] & (1 << k)) >> k;
            uint8_t top_left = 0,
                    top_mid = 0,
                    top_right = 0,
                    mid_left = 0,
                    mid_right = 0,
                    bot_left = 0,
                    bot_mid = 0,
                    bot_right = 0;
            
            if (i != 0 && j != 0) {
                top_left = k == 0 ?
                    (shmem[(i - 1) * width + (j - 1)] & (1 << 7)) >> 7 :
                    (shmem[(i - 1) * width + j] & (1 << (k - 1))) >> (k - 1);
            }

            if (i != 0) {
                top_mid = (shmem[(i - 1) * width + j] & (1 << k)) >> k;
            }

            if (i != 0 && j != width) {
                top_left = k == 7 ?
                    (shmem[(i - 1) * width + (j + 1)] & (1 << 0)) >> 0 :
                    (shmem[(i - 1) * width + j] & (1 << (k + 1))) >> (k + 1);
            }

            if (j != 0) {
                mid_left = k == 0 ?
                    (shmem[i * width + (j - 1)] & (1 << 7)) >> 7 :
                    (shmem[i * width + j] & (1 << (k - 1))) >> (k - 1);
            }

            if (j != width) {
                mid_right = k == 7 ?
                    (shmem[i * width + (j + 1)] & (1 << 0)) >> 0 :
                    (shmem[i * width + j] & (1 << (k + 1))) >> (k + 1);
            }

            if (i != height && j != 0) {
                bot_left = k == 0 ?
                    (shmem[(i + 1) * width + (j - 1)] & (1 << 7)) >> 7 :
                    (shmem[(i + 1) * width + j] & (1 << (k - 1))) >> (k - 1);
            }

            if (i != height) {
                bot_mid = (shmem[(i + 1) * width + j] & (1 << k)) >> k;
            }

            if (i != height && j != width) {
                bot_right = k == 7 ?
                    (shmem[(i + 1) * width + (j + 1)] & (1 << 0)) >> 0 :
                    (shmem[(i + 1) * width + j] & (1 << (k + 1))) >> (k + 1);
            }

            uint8_t neighbors = top_left + top_mid + top_right + 
                                mid_left +           mid_right + 
                                bot_left + bot_mid + bot_right;

            // Any live cell with two or three neighbors survives.
            if ((neighbors == 2 || neighbors == 3) && current == 1) {
                updated_cells[tidy * width + tidx] |= 1 << (7 - k);
            }
            // Any dead cell with three live neighbors comes to life.
            else if (neighbors == 3 && current == 0) {
                updated_cells[tidy * width + tidx] |= 1 << (7 - k);
            }
            // Any other cells die.
            else {
                updated_cells[tidy * width + tidx] |= 0 << (7 - k);
            }
        }
    }
}

void call_cuda_gol_update(int num_threads,
                          int width, int height,
                          uint8_t* cells, uint8_t* updated_cells,
                          bool optimized) {
    int actual_width = width % 8 == 0 ? width / 8 : width;
    int x_blocks = int((actual_width + num_threads - 1) / num_threads);
    int y_blocks = int((height + num_threads - 1) / num_threads);

    dim3 block_size(num_threads, num_threads);
    dim3 grid_size(x_blocks, y_blocks);

    if (optimized) {
        if (width % 8 == 0) {
            optimized_update_kernel_bitwise<<<grid_size, block_size, 
                (actual_width * (height + 1)) * sizeof(uint8_t)>>>(width, height, 
                cells, updated_cells);
        }
        else {
            optimized_update_kernel<<<grid_size, block_size, 
                (width * (height + 1)) * sizeof(uint8_t)>>>(width, height, 
                cells, updated_cells);
        }
    }
    else {
        naive_update_kernel<<<grid_size, block_size>>>(width, height, 
            cells, updated_cells);
    }
}