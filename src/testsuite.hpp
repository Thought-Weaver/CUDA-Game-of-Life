/**
 * Header for testsuite.
 * @author Logan Apple
 * @date 5/18/2020
 */

#ifndef TESTS_H
#define TESTS_H

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "grid.hpp"
#include "utils.hpp"

#define num_blocks 16

/* Run a series of comprehensive tests on IO utilities. */
void run_io_tests();

/* Run a series of comprehensive tests on other Grid functions. */
void run_other_grid_tests();

/* Run a full suite of tests on a grid using all the update methods. */
void run_grid_update_tests();

#endif