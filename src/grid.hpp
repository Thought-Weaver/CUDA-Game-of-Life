/**
 * Header for the Grid class, used to store the cellular automaton.
 * @author Logan Apple
 * @date 5/13/2020
 */

#ifndef GRID_H
#define GRID_H

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <stdint.h>
#include <time.h>

class Grid {
    private:
        /* A 2D array of size width x height containing a 0 in index i, j if
         * the cell is dead and a 1 if it's alive. This is the current state
         * of the grid.
         */
        uint8_t* cells;

        /* A 2D array of size width x height containing a 0 in index i, j if
         * the cell is dead and a 1 if it's alive. This is the current state
         * of the grid.
         */
        uint8_t* bitwise_cells;

        /* The width and height of the grid. */
        int width, height;

        /* Counts the living neighbors of a cell. */
        int count_neighbors(int x, int y);

    public:
        /* The constructor for Grid. */
        Grid(int width, int height, uint8_t* initial_state);

        /* The destructor for Grid. */
        ~Grid();

        /* Get the current cell state. */
        uint8_t* get_cells();

        /* Sets cells to another state of cells. */
        void set_cells(uint8_t* other_cells);

        /* Convert the current cells to a bitwise form. */
        uint8_t* convert_to_bitwise();

        /* Set current cells to converted bitwise cells in regular form. */
        uint8_t* convert_to_regular();

        /* Update the current cells to the next state using a naive CPU method. 
         */
        void naive_cpu_update();

        /* Update the current cells to the next state using a naive GPU method. 
         */
        void naive_gpu_update(int blocks);

        /* Update the current cells to the next state using an optimized GPU 
         * method. 
         */
        void optimized_gpu_update(int blocks);

        /* Operator overload for comparing two grids. */
        bool operator==(const Grid& g) {
            if (g.width != width || g.height != height) {
                return false;
            }

            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    if (cells[i * width + j] != g.cells[i * width + j]) {
                        return false;
                    }
                }
            }

            return true;
        }
};

#endif