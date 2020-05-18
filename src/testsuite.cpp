/**
 * A series of rigorous tests for the cellular automaton.
 * @author Logan Apple
 * @date 5/18/2020
 */

#include "testsuite.hpp"

using dir_it = std::filesystem::directory_iterator;

/* Checks to see if two cell states are equal. */
bool check_equal(int width, int height, int* cells, int* other_cells) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (cells[i * width + j] != other_cells[i * width + j]) {
                return false;
            }
        }
    }
    return true;
}

/* Run a series of comprehensive tests on IO utilities. */
void run_io_tests() {
    int* result_1 = new int[10 * 10] {
        0,0,1,0,0,0,0,0,0,0,
        1,0,1,0,0,0,0,0,0,0,
        0,1,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    };

    int* result_2 = new int[17 * 15] {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,
        0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,
        0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,
        0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,
        0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,
        0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,
        0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };

    int* result_3 = new int[30 * 20] {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,
        1,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,0,0,1,0,0,0,
        0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,
        0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
        0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,
        0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,
        0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };

    int* glider = load_cells(10, 10, "./grids/10x10_Glider.txt");
    int* pulsar = load_cells(17, 15, "./grids/17x15_Pulsar.txt");
    int* puffer = load_cells(30, 20, "./grids/30x20_Puffer1.txt");

    // All checks for load_cells.
    assert(check_equal(10, 10, glider, result_1));
    assert(check_equal(17, 16, pulsar, result_2));
    assert(check_equal(30, 20, puffer, result_3));

    // Output check, just to make sure it exists.
    output_frame(10, 10, "test", glider, 0);
    std::ifstream file("./output_frames/test_0.ppm");
    assert(file.good());

    // Free memory.
    delete[] result_1;
    delete[] result_2;
    delete[] result_3;
    delete[] glider;
    delete[] pulsar;
    delete[] puffer;
}

/* Run a series of comprehensive tests on other Grid functions. */
void run_other_grid_tests() {
    int* test_state_1 = new int[3 * 4] {
        0, 0, 0,
        1, 0, 1,
        0, 1, 0
    };

    int* test_state_2 = new int[3 * 4] {
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
    };

    Grid* grid = new Grid(3, 4, test_state_1);

    // Make sure get_cells works.
    assert(check_equal(3, 4, grid->get_cells(), test_state_1));

    // Make sure set_cells works.
    grid->set_cells(test_state_2);
    assert(check_equal(3, 4, grid->get_cells(), test_state_2));

    // Free memory.
    delete[] test_state_1;
    delete[] test_state_2;
    delete grid;
}

/* Run a series of comprehensive tests on Grid update methods. */
void run_grid_update_tests() {
    // Iterate over all of the tests in the folder.
    for (const auto& p : dir_it("./tests/inputs")) {
        // Temp string for storing substrings split by delimeter.
        std::string s;
        // String of path to current test file.
        std::string path_str = std::string(p.path().u8string());
        // String of base test filename.
        std::string base_filename = 
            path_str.substr(path_str.find_last_of("/\\") + 1);
        // Stringstream for splitting the filename.
        std::stringstream path_ss = std::stringstream(base_filename);
        // Resulting vector of substrings split by delimeter.
        std::vector<std::string> split_str;

        while(std::getline(path_ss, s, '_')) {
            split_str.push_back(s);
        }

        // Get params from split filename.
        int width = atoi(split_str[0].c_str());
        int height = atoi(split_str[1].c_str());
        int iterations = atoi(split_str[2].c_str());

        int* initial_state = load_cells(width, height, path_str);
        int* solution_state = load_cells(width, height, 
                                         "./tests/solutions/" + base_filename);

        Grid* grid = new Grid(width, height, initial_state);

        // Iterates the specified number of times on the CPU method.
        for (int i = 0; i < iterations; ++i) {
            grid->naive_cpu_update();
        }

        // Make sure solution matches resulting grid.
        assert(check_equal(width, height, grid->get_cells(), solution_state));

        // Reset grid to initial state for next method test.
        grid->set_cells(initial_state);

        // Iterates the specified number of times on the naive GPU method.
        for (int i = 0; i < iterations; ++i) {
            grid->naive_gpu_update(num_blocks);
        }

        // Make sure solution matches resulting grid.
        assert(check_equal(width, height, grid->get_cells(), solution_state));

        // TODO: Add test for optimized GPU update when ready.

        // Free memory.
        delete[] initial_state;
        delete[] solution_state;
        delete grid;
    }
}