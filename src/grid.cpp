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

/* Convert the current cells to a bitwise form. */
uint8_t* Grid::convert_to_bitwise() {
    int bitwise_width = width / 8;
    uint8_t converted[] = new uint8_t[bitwise_width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < bitwise_width; ++j) {
            // Use bitwise operations to store 8 cells in a single position.
            for (int k = 0; k < 8; ++k) {
                converted[i * bitwise_width + j] |= 
                    cells[i * width + (j * 8 + k)] << (7 - k);
            }
        }
    }

    return converted;
}

/* Set current cells to converted bitwise cells in regular form. */
void Grid::convert_to_regular(uint8_t* bitwise_cells) {
    int bitwise_width = width / 8;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < bitwise_width; ++j) {
            for (int k = 0; k < 8; ++k) {
                cells[i * width + (j * 8 + k)] =
                    (cells[i * bitwise_width + j] & (1 << k)) >> k;
            }
        }
    }
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
    call_cuda_gol_update(blocks, 
                         width, height,
                         dev_cells, dev_out_cells, false);
    
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
void Grid::optimized_gpu_update(int blocks) {
    uint8_t* dev_cells;
    uint8_t* dev_out_cells;

    int actual_width = width % 8 == 0 ? width / 8 : width;

    // Allocate memory for GPU computation.
    gpuErrchk(cudaMalloc((void **) &dev_cells, 
        actual_width * height * sizeof(uint8_t)));
    gpuErrchk(cudaMalloc((void **) &dev_out_cells, 
        actual_width * height * sizeof(uint8_t)));

    // Copy memory to device.
    if (width % 8 == 0) {
        // In this special case, we can optimize further.
        gpuErrchk(cudaMemset(dev_out_cells, 0, 
            actual_width * height * sizeof(uint8_t)));
        gpuErrchk(cudaMemcpy(dev_cells, convert_to_bitwise(cells), 
            actual_width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
    else {
        gpuErrchk(cudaMemcpy(dev_cells, cells, 
            actual_width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    // Update the cells using optimized GPU method.
    call_cuda_gol_update(blocks, 
                         width, height,
                         dev_cells, dev_out_cells, true);
    
    uint8_t* updated_cells = new uint8_t[actual_width * height];

    // Copy memory back to host.
    if (width % 8 == 0) {
        gpuErrchk(cudaMemcpy(updated_cells, dev_out_cells, 
            actual_width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        convert_to_regular(updated_cells);

        delete[] updated_cells;
    }
    else {
        gpuErrchk(cudaMemcpy(updated_cells, dev_out_cells, 
            actual_width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        // Now that the next generation has been computed in updated_cells,
        // copy that to the cells in the Grid object.
        set_cells(updated_cells);
        
        delete[] updated_cells;
    }

    // Free memory.
    cudaFree(dev_cells);
    cudaFree(dev_out_cells);
}