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

/* Takes a state of cells and outputs it to a PPM file. */
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

/* Takes a state of cells and outputs it to standard output. */
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

/* Loads a state of cells from a file. */
int* load_cells(int width, int height, std::string filename) {

}

/* Checks to see that the input arguments are valid. */
void check_args(int argc, char **argv) {
    #if GPU
        if (argc < 6) {
            std::cerr << "Incorrect number of arguments." << std::endl;
            std::cerr << "Usage: cpu-gol {width} {height} {iterations} "
                    << "{threads per block} {max num of blocks} -f " 
                    << "{optional input file}" << std::endl;
            exit(EXIT_FAILURE);
        }
    #else
        if (argc < 4) {
            std::cerr << "Incorrect number of arguments." << std::endl;
            std::cerr << "Usage: cpu-gol {width} {height} {iterations} -f " 
                    << "{optional input file}" << std::endl;
            exit(EXIT_FAILURE);
        }
    #endif
}

int main(int argc, char** argv) {
    // Filename for loading an initial board.
    std::string filename = "";
    // Basic parameters, set to defaults, just in case.
    int iterations = 0, 
        width = 10, 
        height = 10,
        num_threads = 512,
        num_blocks = 200;

    // Make sure all arguments are valid and that the right number is present.
    check_args(argc, argv);

    // Parse command line arguments.
    #if GPU
        width       = atoi(argv[1]);
        height      = atoi(argv[2]);
        iterations  = atoi(argv[3]);
        num_threads = atoi(argv[4]);
        num_blocks  = atoi(argv[5]);
        for (int i = 6; i < argc; ++i) {
            if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-f") == 0) {
                ++i;
                if (i < argc) {
                    filename = argv[i];
                }
            }
        }
    #else
        width      = atoi(argv[1]);
        height     = atoi(argv[2]);
        iterations = atoi(argv[3]);
        for (int i = 4; i < argc; ++i) {
            if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-f") == 0) {
                ++i;
                if (i < argc) {
                    filename = argv[i];
                }
            }
        }
    #endif

    // If no filename was specified, create a random initial state.
    int* initial_state = new int[width * height];
    if (filename == "") {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                initial_state[i * width + j] = rand() % 2;
            }
        }
    }
    else {
        initial_state = load_cells(width, height, filename);
    }

    // Create grid with initial cell state.
    Grid *grid = new Grid(width, height, initial_state);

    // Show initial state.
    print_cells(width, height, grid->get_cells());

    // Update and print grid.
    for (int i = 0; i < iterations; ++i) {
        grid->naive_cpu_update();
        print_cells(width, height, grid->get_cells());
    }

    // Free memory.
    delete[] initial_state;
    delete grid;

    return 0;
}