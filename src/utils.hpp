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

/* Takes a state of cells and outputs it to a PPM file. */
void output_frame(int width, int height, std::string filename, int* cells, 
                  int iter_num);

/* Takes a state of cells and outputs it to standard output. */
void print_cells(int width, int height, int* cells);

/* Loads a state of cells from a file. */
int* load_cells(int width, int height, std::string filename);

#endif