/**
 * Main file for running the cellular automaton.
 * @author Logan Apple
 * @date 5/13/2020
 */

#include "grid.hpp"
#include "utils.hpp"

#if TEST
    #include "testsuite.hpp"
#endif

/* Checks to see that the input arguments are valid. */
void check_args(int argc, char **argv) {
    #if GPU
        if (argc < 5) {
            std::cerr << "Error: Incorrect number of arguments." << std::endl;
            std::cerr << "Usage: gpu-gol {width} {height} {iterations} "
                      << "{num of blocks} -f {optional input file}" 
                      << std::endl;
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
    // If we're in testing mode, run tests and skip everything else.
    #if TEST
        run_io_tests();
        run_other_grid_tests();
        run_grid_update_tests();
        exit(EXIT_SUCCESS);
    #endif

    // Filename for loading an initial board.
    std::string in_filename = "";
    // Base filename for outputting cell frames.
    std::string out_filename = "";
    // Basic parameters, set to defaults, just in case.
    int iterations = 0, 
        width = 10, 
        height = 10,
        num_blocks = 16,
        delay = 100; // in ms
    // Print output boolean option.
    bool quiet = false;

    // Make sure all arguments are valid and that the right number is present.
    // TODO: With optional arguments, this makes checking the count impossible.
    check_args(argc, argv);

    // Parse command line arguments.
    #if GPU
        try {
            width       = std::stoi(argv[1]);
            height      = std::stoi(argv[2]);
            iterations  = std::stoi(argv[3]);
            num_blocks  = std::stoi(argv[4]);
        }
        catch(std::exception const& e) {
            std::cerr << "Error: " << e.what() << std::endl << std::endl;

            std::cerr << "Usage: gpu-gol {width} {height} {iterations} "
                      << "{num of blocks} -f {optional input file}" 
                      << std::endl;
        }

        for (int i = 5; i < argc; ++i) {
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

                ++i;
                try {
                    delay = std::stoi(argv[i]);
                }
                catch(std::exception const& e) {
                    std::cerr << "Error: " << e.what() << std::endl << std::endl;

                    std::cerr << "Usage: gpu-gol {width} {height} {iterations} "
                            << "{num of blocks} -o {base filename} {delay in ms}" 
                            << std::endl;
                }
            }

            if (strcmp(argv[i], "-q") == 0) {
                quiet = true;
            }
        }
    #else
        try {
            width       = std::stoi(argv[1]);
            height      = std::stoi(argv[2]);
            iterations  = std::stoi(argv[3]);
        }
        catch(std::exception const& e) {
            std::cerr << "Error: " << e.what() << std::endl << std::endl;

            std::cerr << "Usage: cpu-gol {width} {height} {iterations} "
                      << "-f {optional input file}" 
                      << std::endl;
        }

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

                ++i;
                try {
                    delay = std::stoi(argv[i]);
                }
                catch(std::exception const& e) {
                    std::cerr << "Error: " << e.what() << std::endl << std::endl;

                    std::cerr << "Usage: gpu-gol {width} {height} {iterations} "
                            << "{num of blocks} -o {base filename} {delay in ms}" 
                            << std::endl;
                }
            }

            if (strcmp(argv[i], "-q") == 0) {
                quiet = true;
            }
        }
    #endif

    // If no filename was specified, create a random initial state.
    uint8_t* initial_state = new uint8_t[width * height];
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

    // Parameters for GIF output.
	GifWriter g;
    GifAnim ganim;
    if (out_filename != "") {
	    ganim.GifBegin(&g, ("./gifs/" + out_filename + ".gif").c_str(), 
                       width, height, delay);
    }

    // Update and print grid.
    for (int i = 0; i < iterations; ++i) {
        #if GPU
            grid->naive_gpu_update(num_blocks);
        #else
            grid->naive_cpu_update();
        #endif

        if (!quiet) {
            print_cells(width, height, grid->get_cells());
        }

        // Write an output frame every iteration.
        if (out_filename != "") {
            //output_frame(width, height, out_filename, grid->get_cells(), i);
            output_gif_frame(width, height, grid->get_cells(), 
                             &g, &ganim, delay);
        }
    }

    // Close and write the GIF.
    if (out_filename != "") {
        ganim.GifEnd(&g);
    }

    // Free memory.
    delete[] initial_state;
    delete grid;

    return 0;
}