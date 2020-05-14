/**
 * Main file for running the cellular automaton.
 * @author Logan Apple
 * @date 5/13/2020
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>

#include "grid.hpp"

// Takes a state of cells and outputs it to a PPM file.
void output_frame(int width, int height, std::string filename, int* cells) {
    std::ofstream output((filename + ".ppm").c_str());
    output << "P3" << std::endl;
    output << width << " " << height << std::endl;
    output << "255" << std::endl;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            if (cells[i * width + j] > 0) {
                int pixel_value = 255 * cells[i * width + j];
                output << pixel_value << " " << 
                          pixel_value << " " << 
                          pixel_value << " " << std::endl;
            }
            else {
                output << "0 0 0" << std::endl;
            }
        }
    }
    output.close();
}

// Takes a state of cells and outputs it to standard output.
void print_cells(int width, int height, int* cells) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            if (cells[i * width + j] == 0) {
                std::cout << ".";
            }
            else {
                std::cout << "#";
            }
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    int width = 10, height = 10;
    int initial_state[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1
    };
    Grid grid = Grid(width, height, initial_state);
    
    print_cells(width, height, grid.get_cells());

    int iterations = 5;
    for (int i = 0; i < iterations; ++i) {
        grid.naive_cpu_update();
        print_cells(width, height, grid.get_cells());
    }

    return 0;
}