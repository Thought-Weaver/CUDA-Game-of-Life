/**
 * Main file for running the cellular automaton.
 * @author Logan Apple
 * @date 5/13/2020
 */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <time.h>

#include "grid.hpp"

/* Takes a state of cells and outputs it to a PPM file. */
void output_frame(int width, int height, std::string filename, int* cells, 
                  int iter_num) {
    std::ofstream output(("./output_frames/" + filename + "_" + 
                          std::to_string(iter_num) + ".ppm").c_str());
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
    // Allocate memory for initial state.
    int* initial_state = new int[width * height];

    std::ifstream file(filename.c_str());

    // Check to see if file exists.
    if (!file.good()) {
        std::cerr << "Error: Input file doesn't exist." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    int i = 0;
    // Iterate over the lines in the file.
    while (std::getline(file, line)) {
        int j = 0;
        
        // Strip newline characters.
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

        // Make sure there aren't more lines than there should be.
        if (i >= height) {
            std::cerr << "Error: There were " << (i + 1) << " or more lines in "
                      << "the input file, but expected " << height << " lines."
                      << std::endl;
            exit(EXIT_FAILURE);
        } 

        // Make sure that the line length is the same as width.
        if (line.length() != width) {
            std::cerr << "Error: Line " << i << " of input file had length "
                      << line.length() << ", but expected a line of length "
                      << width << "." << std::endl;
            exit(EXIT_FAILURE);
        }

        // Iterate over characters in line.
        for (char& c : line) {
            // Quick ASCII hack for converting a char to an int.
            initial_state[i * width + j] = c - '0';
            j += 1;
        }
        i += 1;
    }
    file.close();

    // Make sure there aren't fewer lines than there should be.
    if (i < height) {
        std::cerr << "Error: There were " << i << " lines in the "
                  << "input file, but expected " << height << " lines."
                  << std::endl;
        exit(EXIT_FAILURE);
    } 
    
    return initial_state;
}

/* Checks to see that the input arguments are valid. */
void check_args(int argc, char **argv) {
    #if GPU
        if (argc < 6) {
            std::cerr << "Error: Incorrect number of arguments." << std::endl;
            std::cerr << "Usage: cpu-gol {width} {height} {iterations} "
                      << "{threads per block} {max num of blocks} -f " 
                      << "{optional input file}" << std::endl;
            exit(EXIT_FAILURE);
        }
    #else
        if (argc < 4) {
            std::cerr << "Error: Incorrect number of arguments." << std::endl;
            std::cerr << "Usage: cpu-gol {width} {height} {iterations} -f " 
                      << "{optional input file}" << std::endl;
            exit(EXIT_FAILURE);
        }
    #endif
}

int main(int argc, char** argv) {
    // Filename for loading an initial board.
    std::string in_filename = "";
    // Base filename for outputting cell frames.
    std::string out_filename = "";
    // Basic parameters, set to defaults, just in case.
    int iterations = 0, 
        width = 10, 
        height = 10,
        num_threads = 512,
        num_blocks = 200;
    // Print output boolean option.
    bool quiet = false;

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
                    in_filename = argv[i];
                }
            }
            
            if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "-o") == 0) {
                ++i;
                if (i < argc) {
                    out_filename = argv[i];
                }
            }

            if (strcmp(argv[i], "-q") == 0) {
                quiet = true;
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
                    in_filename = argv[i];
                }
            }

            if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "-o") == 0) {
                ++i;
                if (i < argc) {
                    out_filename = argv[i];
                }
            }

            if (strcmp(argv[i], "-q") == 0) {
                quiet = true;
            }
        }
    #endif

    // If no filename was specified, create a random initial state.
    int* initial_state = new int[width * height];
    if (in_filename == "") {
        // Guarantee random generation.
        srand (time(NULL));
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                initial_state[i * width + j] = rand() % 2;
            }
        }
    }
    else {
        initial_state = load_cells(width, height, in_filename);
    }

    // Create grid with initial cell state.
    Grid *grid = new Grid(width, height, initial_state);

    // Show initial state if not in quiet mode.
    if (!quiet) {
        print_cells(width, height, grid->get_cells());
    }

    // Update and print grid.
    for (int i = 0; i < iterations; ++i) {
        grid->naive_cpu_update();

        if (!quiet) {
            print_cells(width, height, grid->get_cells());
        }

        if (out_filename != "") {
            output_frame(width, height, out_filename, grid->get_cells(), i);
        }
    }

    // Free memory.
    delete[] initial_state;
    delete grid;

    return 0;
}