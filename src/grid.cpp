/**
 * Implemented methods for the Grid class, used to store the cellular automaton.
 * @author Logan Apple
 * @date 5/14/2020
 */

#include "grid.hpp"
#include "gol.cuh"

/* Constructor for the grid. */
Grid::Grid(int w, int h, uint8_t* initial_state) {
    // I should probably add some error check that verifies that initial_state
    // is the correct size.
    width = w;
    height = h;
    cells = new uint8_t[width * height];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cells[i * width + j] = initial_state[i * width + j];
        }
    }
}

/* Destructor for the grid. */
Grid::~Grid() {
    delete[] cells;
}

/* Get the current cell state. */
uint8_t* Grid::get_cells() {
    return cells;
}

/* Sets cells to another state of cells. */
void Grid::set_cells(uint8_t* other_cells) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cells[i * width + j] = other_cells[i * width + j];
        }
    }
}

/* Get the history of cell states. */
uint8_t** Grid::get_history() {
    return history;
}

/* Counts the living neighbors of a cell. */
int Grid::count_neighbors(int x, int y) {
    int alive = 0;
    
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0;
    }

    for (int i = y - 1; i <= y + 1; ++i) {
        for (int j = x - 1; j <= x + 1; ++j) {
            if (i != y || j != x) {
                // Could make this wrap around? Might be a problem for GPU.
                if (i >= 0 && i < height && j >= 0 && j < width) {
                    alive += cells[i * width + j];
                }
            }
        }
    }

    return alive;
}

/* Update the current cells to the next state using a naive CPU method. */
void Grid::naive_cpu_update() {
    // Number of living neighbors any cell has.
    int neighbors = 0;
    // A new array to store the next generation.
    uint8_t* updated_cells = new uint8_t[width * height];
    // Iterate over the cells and apply Conway's rules.
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            neighbors = count_neighbors(j, i);
            // Any live cell with two or three neighbors survives.
            if (cells[i * width + j] == 1 && (neighbors == 2 || neighbors == 3)) {
                updated_cells[i * width + j] = 1;
            }
            // Any dead cell with three live neighbors comes to life.
            else if (cells[i * width + j] == 0 && neighbors == 3) {
                updated_cells[i * width + j] = 1;
            }
            // Any other cells die.
            else {
                updated_cells[i * width + j] = 0;
            }
        }
    }
    
    // Now that the next generation has been computed in updated_cells,
    // copy that to the cells in the Grid object.
    set_cells(updated_cells);

    // Free memory.
    delete[] updated_cells;
}

/* Update the current cells to the next state using a naive GPU method. */
void Grid::naive_gpu_update(int blocks) {
    uint8_t* dev_cells;
    uint8_t* dev_out_cells;

    // Allocate memory for GPU computation.
    gpuErrchk(cudaMalloc((void **) &dev_cells, 
        width * height * sizeof(uint8_t)));
    gpuErrchk(cudaMalloc((void **) &dev_out_cells, 
        width * height * sizeof(uint8_t)));

    // Copy memory to device.
    gpuErrchk(cudaMemcpy(dev_cells, cells, 
        width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_out_cells, 0, 
        width * height * sizeof(uint8_t)));

    // Update the cells using naive GPU method.
    call_cuda_naive_gol_update(blocks, 
        width, height, 
        dev_cells, dev_out_cells);
    
    // Copy memory back to host.
    uint8_t* updated_cells = new uint8_t[width * height];
    gpuErrchk(cudaMemcpy(updated_cells, dev_out_cells, 
            width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    // Now that the next generation has been computed in updated_cells,
    // copy that to the cells in the Grid object.
    set_cells(updated_cells);

    // Free memory.
    cudaFree(dev_cells);
    cudaFree(dev_out_cells);

    delete[] updated_cells;
}

/* Update the current cells to the next state using an optimized GPU method. */
void Grid::optimized_gpu_update(int blocks, int iterations) {
    uint8_t** dev_history;
    history = new uint8_t*[iterations + 1];

    // In order to copy over an array of pointers, we need to start by making
    // a host array of pointers to device arrays, then copy it into a device
    // array of pointers to device arrays.
    for (int i = 0; i <= iterations; ++i) {
        gpuErrchk(cudaMalloc((void **) &history[i], 
            width * height * sizeof(uint8_t)));
    }

    // Copy memory to device.
    gpuErrchk(cudaMemcpy(history[0], cells, 
        width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Allocate history.
    gpuErrchk(cudaMalloc(&dev_history, 
        (iterations + 1) * sizeof(uint8_t*)));

    // Copy host history to device history.
    gpuErrchk(cudaMemcpy(dev_history, history, 
        (iterations + 1) * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    for (int i = 0; i <= iterations; ++i) {
        gpuErrchk(cudaMemcpy(dev_history[i], history[i], 
            width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    // Update the cells using optimized GPU method.
    call_cuda_opt_gol_update(blocks, 
                             width, height,
                             iterations, dev_history);

    // Copy memory back to host.
    gpuErrchk(cudaMemcpy(history, dev_history, 
        sizeof(uint8_t*) * iterations, cudaMemcpyDeviceToHost));

    // Free memory.
    for (int i = 0; i <= iterations; ++i) {
        cudaFree(dev_history[i]);
    }
    cudaFree(dev_history);
}