# CS 179 Final Project: Parallelizing Conway's Game of Life

## Summary and Motivation

Conway's Game of Life is a classic cellular automaton with rules that allow for numerous discrete lifeforms. While small-scale simulations run perfectly fine in a serial implementation of the associated algorithms, those on a larger scale require more clever implementations. The goal of this project was to parallelize the algorithms involved with computing the Game of Life using CUDA.

The Game of Life is a cellular automaton in which cells have two states, being dead or alive, and evolve on a two-dimensional grid according to the following rules, which are evaluated on each cell with respect to its eight neighboring cells:

1. If the cell is alive and has less than two or greater than three living neighbors, then the cell dies.
2. If the cell is alive and has exactly two or three living neighbors, it remains alive.
3. If the cell is dead and has exactly three living neighbors, then the cell lives.

In the parallelized implementation, we compute this naively using global memory and in a more optimized fashion using shared memory. If the width of the grid is mod 8, then we can compress the grid into a special format where each cell is represented by a single bit rather than a byte. In this case, we simply perform the update using bitwise operations.

A tribute to John Conway (1937-2020).

## Code Structure

Here's a quick overview of the structure of the project:

* The "bin" directory contains the output of running the makefile.
* The "gifs" directory contains the output GIFs, if any, when running the program.
* The "grids" directory contains some pre-made grids that can be loaded into the program.
* The "obj" directory contains the intermediate output of running the makefile.
* The "src" directory contains the primary sourcefiles for the project.
* The "tests" directory contains inputs and solutions that are used for the testsuite.

The source files are used as follows:

* The "gifanim" files are used to generate GIF frames.
* The "gol" files have the CUDA kernels for parallel processing the Game of Life updates.
* The "grid" files have the class structure for the grid that stores the Game of Life and the algorithms for CPU updates.
* The "main" file runs the program and switches between modes depending on how the source is compiled.
* The "testsuite" files store the tests for the program that are run when compiled for testing.
* The "utils" files have some I/O utilities for loading grids and saving GIFs.

## Usage

There are two main ways to build the project:

```
make cpu-gol
make gpu-gol
```

These will build a version of the project that uses the CPU implementation and GPU implementation of the algorithms, respectively. You can run them by using:

```
./bin/cpu-gol {width} {height} {iter}
./bin/gpu-gol {width} {height} {iter} {threads per block} {-opt for optimized}
```

Both of these will run the Game of Life with randomly generated boards. There are other, optional parameters you can specify:

-f {filename} will load the input file of 0s and 1s and convert it into a cell grid.

-o {base filename} {gif frame delay in hundredths of seconds} will generate an output GIF with the given frame delay as ./gifs/{base filename}.gif

-q will enable quiet mode and prevent printing the cell state to the terminal.

-opt will enable the optimized GPU mode.

Some example grids you can load with -f are available in the grids directory.

Here's an example of running the program on the CPU with a period-3 pulsar:

```
./bin/cpu-gol 17 15 6 -f ./grids/17x15_Pulsar.txt
````

Here's an example of running the optimized GPU version on a 512x512 board and outputting it to ./gifs/random.gif with a 0.1 second frame delay:

```
./bin/gpu-gol 512 512 1000 16 -opt -o random 10
```

## Example Output

Here's an example running the program on 128x128_Random.txt in grids:

![Example Output](gifs/example.gif)

## Tests

You can also build the project using make test. This will create an executable called test in the bin directory. Running ./bin/test will run a series of tests on the IO and Grid class functions. If everything runs fine, the program will print out numerous successes; if not, it will exit with an assertion error.

## Optimizations

* Optimized form uses shared memory.
* Used a y-dimension for block size > 1.
* Changed from int to uint_8 which is 8 bits instead of 32.
* Unrolled loops for counting neighbors.
* Using short-circuiting on the GoL rules to avoid unnecessary accesses.
* Removed memset to reduce number of repeated writes to GPU memory.
* If width is divisible by 8, updates are computed on the bitwise form instead.

## Discussion

I tested a variety of optimizations, all of which are visible in the git commit history. Texture memory, in particular, didn't turn out to be sufficiently fast to keep around. As far as I can tell, the primary bottleneck with the program is actually I/O; copying back and forth from the host to the device and so forth appears to be the largest source of delays.

See the following brainstorming document for more discussion:

https://bit.ly/3bZXaE0