/**
 * IO utilities for loading cells, printing, and frame output.
 * @author Logan Apple
 * @date 5/18/2020
 */

#include "utils.hpp"

/* Takes a vector of cells and outputs it into a GIF. */
void output_gif(int width, int height, std::string filename, 
                std::vector<uint8_t*> history) {
	int delay = 100;
	GifWriter g;
	GifBegin(&g, (filename + ".gif").c_str(), width, height, delay);
    for (auto& cells : history) {
        GifWriteFrame(&g, cells, width, height, delay);
    }
	GifEnd(&g);
}


/* Takes a state of cells and outputs it to a PPM file. */
void output_frame(int width, int height, std::string filename, uint8_t* cells, 
                  int iter_num) {
    std::ofstream output(("./output_frames/" + filename + "_" + 
                          std::to_string(iter_num) + ".ppm").c_str());
    output << "P3" << std::endl;
    output << width << " " << height << std::endl;
    output << "255" << std::endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (cells[i * width + j] > 0) {
                uint8_t pixel_value = 255 * cells[i * width + j];
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
void print_cells(int width, int height, uint8_t* cells) {
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
uint8_t* load_cells(int width, int height, std::string filename) {
    // Allocate memory for initial state.
    uint8_t* initial_state = new uint8_t[width * height];

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