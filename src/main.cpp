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
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
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
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (cells[i * width + j] == 0) {
                std::cout << ".";
            }
            else {
                std::cout << "#";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Filename for loading an initial board.
    std::string filename = "";

    // Parse command line arguments.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dir") == 0 || strcmp(argv[i], "-d") == 0) {
            ++i;
            if (i < argc) {
                filename = argv[i];
            }
        }
    }

    int width = 10, height = 10;
    int* initial_state = new int[width * height] 
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    Grid *grid = new Grid(width, height, initial_state);

    print_cells(width, height, grid->get_cells());

    int iterations = 5;
    for (int i = 0; i < iterations; ++i) {
        grid->naive_cpu_update();
        print_cells(width, height, grid->get_cells());
    }

    delete[] initial_state;
    delete grid;

    return 0;
}