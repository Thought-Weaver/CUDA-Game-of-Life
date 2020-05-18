/**
 * A series of rigorous tests for the cellular automaton.
 * @author Logan Apple
 * @date 5/18/2020
 */

#include "testsuite.hpp"

/* Checks to see if two cell states are equal. */
bool check_equal(int width, int height, uint8_t* cells, uint8_t* other_cells) {
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
    uint8_t* result_1 = new uint8_t[10 * 10] {
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

    uint8_t* result_2 = new uint8_t[17 * 15] {
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

    uint8_t* result_3 = new uint8_t[30 * 20] {
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

    uint8_t* glider = load_cells(10, 10, "./grids/10x10_Glider.txt");
    uint8_t* pulsar = load_cells(17, 15, "./grids/17x15_Pulsar.txt");
    uint8_t* puffer = load_cells(30, 20, "./grids/30x20_Puffer1.txt");
    
    std::cout << "=================================" << std::endl
              << "IO TESTS:" << std::endl
              << "=================================" << std::endl << std::endl;

    // All checks for load_cells.
    assert(check_equal(10, 10, glider, result_1));

    std::cout << "glider input: PASSED" << std::endl;

    assert(check_equal(17, 16, pulsar, result_2));

    std::cout << "pulsar input: PASSED" << std::endl;

    assert(check_equal(30, 20, puffer, result_3));

    std::cout << "puffer input: PASSED" << std::endl;

    // Output check, just to make sure it exists.
    output_gif(10, 10, "test", std::vector<uint8_t*> { glider });
    std::ifstream file("./gifs/test.gif");
    assert(file.good());

    std::cout << "output: PASSED" << std::endl << std::endl;

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
    uint8_t* test_state_1 = new uint8_t[3 * 4] {
        0, 0, 0,
        1, 0, 1,
        0, 1, 0
    };

    uint8_t* test_state_2 = new uint8_t[3 * 4] {
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
    };

    Grid* grid = new Grid(3, 4, test_state_1);

    std::cout << "=================================" << std::endl
              << "OTHER GRID TESTS:" << std::endl
              << "=================================" << std::endl << std::endl;

    // Make sure get_cells works.
    assert(check_equal(3, 4, grid->get_cells(), test_state_1));

    std::cout << "get_cells: PASSED" << std::endl;

    // Make sure set_cells works.
    grid->set_cells(test_state_2);
    assert(check_equal(3, 4, grid->get_cells(), test_state_2));

    std::cout << "set_cells: PASSED" << std::endl << std::endl;

    // Free memory.
    delete[] test_state_1;
    delete[] test_state_2;
    delete grid;
}

/* Run a series of comprehensive tests on Grid update methods. */
void run_grid_update_tests() {
    // Wanted to use filesystem, but Titan doesn't have C++17. :(
    std::string base_files[4] = {
        "10_10_1_Glider.txt",
        "10_10_2_Glider.txt",
        "10_10_3_Glider.txt",
        "10_10_4_Glider.txt"
    };

    std::cout << "=================================" << std::endl
              << "GRID UPDATE TESTS:" << std::endl
              << "=================================" << std::endl << std::endl;

    // Iterate over all of the tests in the folder.
    for (const auto& base_file : base_files) {
        // Temp string for storing substrings split by delimeter.
        std::string s;
        // Stringstream for splitting the filename.
        std::stringstream path_ss = std::stringstream(base_file);
        // Resulting vector of substrings split by delimeter.
        std::vector<std::string> split_str;

        while(std::getline(path_ss, s, '_')) {
            split_str.push_back(s);
        }

        // Get params from split filename.
        int width = atoi(split_str[0].c_str());
        int height = atoi(split_str[1].c_str());
        int iterations = atoi(split_str[2].c_str());

        uint8_t* initial_state = load_cells(width, height, 
                                        "./tests/inputs/" + base_file);
        uint8_t* solution_state = load_cells(width, height, 
                                         "./tests/solutions/" + base_file);

        Grid* grid = new Grid(width, height, initial_state);

        std::cout << base_file << " TEST:" << std::endl << std::endl;

        // Iterates the specified number of times on the CPU method.
        for (int i = 0; i < iterations; ++i) {
            grid->naive_cpu_update();
        }

        // Make sure solution matches resulting grid.
        assert(check_equal(width, height, grid->get_cells(), solution_state));

        std::cout << "CPU: PASSED" << std::endl;

        // Reset grid to initial state for next method test.
        grid->set_cells(initial_state);

        // Iterates the specified number of times on the naive GPU method.
        for (int i = 0; i < iterations; ++i) {
            grid->naive_gpu_update(test_blocks);
        }

        // Make sure solution matches resulting grid.
        assert(check_equal(width, height, grid->get_cells(), solution_state));

        std::cout << "NAIVE GPU: PASSED" << std::endl << std::endl;

        // TODO: Add test for optimized GPU update when ready.

        // Free memory.
        delete[] initial_state;
        delete[] solution_state;
        delete grid;
    }
}