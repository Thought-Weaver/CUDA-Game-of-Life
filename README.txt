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

./bin/gpu-gol {width} {height} {iter} {threads per block} {max num of blocks}

Both of these will run the Game of Life with randomly generated boards. There
are other, optional parameters you can specify:

-f {filename} will load the input file of 0s and 1s and convert it into a cell
grid.

-o {base filename} {gif frame delay in ms} will generate an output GIF with the
given frame delay as ./gifs/{base filename}.gif

-q will enable quiet mode and prevent printing the cell state to the terminal.

Some example grids you can load with -f are available in the ./grids directory.

Here's an example of running the program on the CPU with a period-3 pulsar:

./bin/cpu-gol 17 15 6 -f ./grids/17x15_Pulsar.txt

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

Still a work-in-progress.
