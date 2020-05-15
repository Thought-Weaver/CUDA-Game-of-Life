/**
 * Implemented methods for the Grid class, used to store the cellular automaton.
 * @author Logan Apple
 * @date 5/14/2020
 */

#include "grid.hpp"

/* Constructor for the grid. */
Grid::Grid(int w, int h, int* initial_state) {
    // I should probably add some error check that verifies that initial_state
    // is the correct size.
    width = w;
    height = h;
    cells = new int[width * height];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cells[i * width + j] = initial_state[i * width + j];
        }
    }
}

/* Destructor for the grid. */
Grid::~Grid() {

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
    int* updated_cells = new int[width * height];
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
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cells[i * width + j] = updated_cells[i * width + j];
        }
    }
}

/* Update the current cells to the next state using a naive GPU method. */
void Grid::naive_gpu_update() {

}

/* Update the current cells to the next state using an optimized GPU method. */
void Grid::optimized_gpu_update() {

}

/* Get the current cell state. */
int* Grid::get_cells() {
    return cells;
}