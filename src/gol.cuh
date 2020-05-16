/**
 * CUDA header for running parallel cellular automaton.
 * @author Logan Apple
 * @date 5/15/2020
 */

#ifndef GOL_DEVICE_CUH
#define GOL_DEVICE_CUH

#include <cstdio>
#include <cstdlib>

/*
 * NOTE: You can use this macro to easily check cuda error codes 
 * and get more information. 
 * 
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *   what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line,
                    bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void cuda_call_naive_gol_update_kernel();

void cuda_call_optimized_gol_update_kernel();

#endif