/**
 * Header for IO utilities.
 * @author Logan Apple
 * @date 5/18/2020
 */

#ifndef UTILS_H
#define UTILS_H

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

#include "gifanim.h"

/* Takes an array of cells and writes it as a GIF frame. */
void output_gif_frame(int width, int height, uint8_t* cells, 
                      GifWriter* g, GifAnim* ganim, int delay);

/* Takes a vector of cells and outputs it into a GIF. */
void output_gif(int width, int height, std::string filename, 
                uint8_t** history, int iterations);

/* Takes a state of cells and outputs it to a PPM file. */
void output_frame(int width, int height, std::string filename, uint8_t* cells, 
                  int iter_num);

/* Takes a state of cells and outputs it to standard output. */
void print_cells(int width, int height, uint8_t* cells);

/* Loads a state of cells from a file. */
uint8_t* load_cells(int width, int height, std::string filename);

#endif