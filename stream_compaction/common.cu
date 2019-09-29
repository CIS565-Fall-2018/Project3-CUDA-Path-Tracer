#include "common.h"

void checkCUDAErrorFn_SC(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    while (true);
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int paddedN, int *bools, PathSegment **idata) {
          // get index first and reject if greater than paddedN
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= paddedN)
          {
            return;
          }
 
          // determine if you're a boolean (if you're in the part that's just padded on, give yourself a 0)
          bools[index] = (index < n && idata[index]->remainingBounces > 0 ) ? 1 : 0;
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, PathSegment **odata, PathSegment **idata, const int *bools, const int *indices) {
          
          // get index first
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= n)
          {
            return;
          }

          if (bools[index])
          {
            odata[indices[index]] = idata[index];
          }
        }
    }
}
