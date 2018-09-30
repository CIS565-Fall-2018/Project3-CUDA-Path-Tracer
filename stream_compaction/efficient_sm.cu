#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient_sm.h"

namespace StreamCompaction {
    namespace EfficientSM {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void kernEfficientScan(int N, int *odata, int *idata){
            extern __shared__ int tmp[];
            int index = threadIdx.x;
            if (index >= N) return;

            int offset = 1;
            tmp[2 * index] = idata[2 * index];
            tmp[2 * index + 1] = idata[2 * index + 1];
            // up sweep
            for (int d = (N >> 1); d > 0; d >>= 1){
                __syncthreads();
                if (index < d) tmp[offset * (2 * index + 2) - 1] += tmp[offset * (2 * index + 1) - 1];
                offset <<= 1;
            }
            // clear last digit
            if (index == 0) tmp[N - 1] = 0;
            // down sweep
            for (int d = 1; d < N; d <<= 1){
                offset >>= 1;
                __syncthreads();
                if (index < d){
                    int t = tmp[offset * (2 * index + 1) - 1];
                    tmp[offset * (2 * index + 1) - 1] = tmp[offset * (2 * index + 2) - 1];
                    tmp[offset * (2 * index + 2) - 1] += t;
                }
            }
            __syncthreads();

            odata[2 * index] = tmp[2 * index];
            odata[2 * index + 1] = tmp[2 * index + 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int N = 1 << ilog2ceil(n);
            dim3 fullBlockPerGrid((N + blockSize - 1) / blockSize);
            int* dev_in, *dev_out;

            cudaMalloc((void**) &dev_in, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_in failed");

            cudaMalloc((void**) &dev_out, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_out failed");

            cudaMemset(dev_out, 0, sizeof(int) * N);
            checkCUDAError("cuda Memset failed");

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy HostToDevice failed");

            timer().startGpuTimer();

            kernEfficientScan <<< fullBlockPerGrid, blockSize, 2 * N * sizeof(int) >>> (N, dev_out, dev_in);
            checkCUDAError("kernNaiveScan dev_in failed");


            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy DeviceToHost failed");

            cudaFree(dev_in);
            cudaFree(dev_out);

        }
    }
}
