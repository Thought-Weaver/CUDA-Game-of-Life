/**
 * Header for the Grid class, used to store the cellular automaton.
 * @author Logan Apple
 * @date 5/13/2020
 */

#ifndef GRID_H
#define GRID_H

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>

class Grid {
    private:
        /* List of previous grid states. */
        // Note to self: Does it make sense to keep a history? Why not output
        // a PPM of the grid after every computation and save ourselves the
        // memory? It's potentially good for interactive visualization, but
        // this project likely won't support that. I could argue the case for
        // future development, though.
        std::vector<int*> history;

        /* A 2D array of size width x height containing a 0 in index i, j if
         * the cell is dead and a 1 if it's alive. This is the current state
         * of the grid.
         */
        int* cells;

        /* The width and height of the grid. */
        int width, height;

        /* Counts the living neighbors of a cell. */
        int count_neighbors(int x, int y);

    public:
        /* The constructor for Grid. */
        Grid(int width, int height, int* initial_state);

        /* The destructor for Grid. */
        ~Grid();

        /* Update the current cells to the next state using a naive CPU method. 
         */
        void naive_cpu_update();

        /* Update the current cells to the next state using a naive GPU method. 
         */
        void naive_gpu_update(int blocks, int threads_per_block);

        /* Update the current cells to the next state using an optimized GPU 
         * method. 
         */
        void optimized_gpu_update(int blocks, int threads_per_block);

        /* Get the current cell state. */
        int* get_cells();
};

#endif