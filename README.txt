#=============================================================================#
# CS 179 Final Project: Parallelizing Conway's Game of Life                   #
#=============================================================================#

Created by Logan Apple.

#-----------------------------------------------------------------------------#
# SUMMARY                                                                     #
#-----------------------------------------------------------------------------#

Conway's Game of Life is a classic cellular automaton with rules that allow for
numerous discrete lifeforms. While small-scale simulations run perfectly fine
in a serial implementation of the associated algorithms, those on a larger
scale require more clever implementations. The goal of this project was to
parallelize the algorithms involved with computing the Game of Life using CUDA.

#-----------------------------------------------------------------------------#
# USAGE                                                                       #
#-----------------------------------------------------------------------------#

There are two main ways to build the project:

make cpu-gol
make gpu-gol

These will build a version of the project that uses the CPU implementation and
GPU implementation of the algorithms, respectively. You can run them by using:

./bin/cpu-gol {width} {height} {iter}

./bin/gpu-gol {width} {height} {iter} {threads per block} {-opt for optimized}

Both of these will run the Game of Life with randomly generated boards. There
are other, optional parameters you can specify:

-f {filename} will load the input file of 0s and 1s and convert it into a cell
grid.

-o {base filename} {gif frame delay in hundredths of seconds} will generate an 
output GIF with the given frame delay as ./gifs/{base filename}.gif

-q will enable quiet mode and prevent printing the cell state to the terminal.

-opt will enable the optimized GPU mode.

Some example grids you can load with -f are available in the ./grids directory.

Here's an example of running the program on the CPU with a period-3 pulsar:

./bin/cpu-gol 17 15 6 -f ./grids/17x15_Pulsar.txt

Here's an example of running the optimized GPU version on a 512x512 board and
outputting it to ./gifs/random.gif with a 0.1 second frame delay:

./bin/gpu-gol 512 512 1000 16 -opt -o random 10

#-----------------------------------------------------------------------------#
# TESTS                                                                       #
#-----------------------------------------------------------------------------#

You can also build the project using make test. This will create an executable
called test in the bin directory. Running ./bin/test will run a series of tests
on the IO and Grid class functions. If everything runs fine, the program will
print out numerous successes; if not, it will exit with an assertion error.

#-----------------------------------------------------------------------------#
# OPTIMIZATIONS                                                               #
#-----------------------------------------------------------------------------#

-Optimized form uses shared memory.
-Changed from int to uint_8 which is 8 bits instead of 32.
-Unrolling loops for counting neighbors.
-Using short-circuiting on the GoL rules to avoid unnecessary accesses.
-Removed "all other cells die" rule in the optimized GPU version, since we
 memset to 0 before copying the data to the GPU.

#-----------------------------------------------------------------------------#
# DISCUSSION                                                                  #
#-----------------------------------------------------------------------------#

After several practical tests, I believe that shared memory is actually a worse
option compared to global memory. Since compute compatibility >2 support, the
benefits haven't been that great and we have memory limits with shared memory;
global memory was able to compute a 512x512 board for 10000 iterations without
any problem, but shared memory was unable to allocate enough memory. 