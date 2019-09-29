#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 64

int* dev_gpuScanBuf;
int* dev_idata;

__global__ void kernNaiveScan(int n, int twoToPowerDMinusOne, int* odata, int* idata)
{
  // get index first
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= n)
  {
    return;
  }
  
  // then add the two numbers and put them into the global output buffer
  if (index >= twoToPowerDMinusOne)
  {
    int one = idata[index - twoToPowerDMinusOne];
    int two = idata[index];
    int onePlusTwo = one + two;
    odata[index] = onePlusTwo;
  }
  else
  {
    odata[index] = idata[index];
  }
}

__global__ void kernShiftScan(int n, int* odata, int* idata)
{

  // if your thread index is 0, insert a 0, otherwise everyone else do their own index - 1 in the data array
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= n)
  {
    return;
  }

  if (index == 0)
  {
    odata[index] = 0;
  }
  else
  {
    odata[index] = idata[index - 1];
  }
}

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

          dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

          int nNextHighestPowTwo = 1 << ilog2ceil(n);

          cudaMalloc((void**)&dev_gpuScanBuf, nNextHighestPowTwo * sizeof(int));
          checkCUDAError_SC("cudaMalloc buf failed");

          cudaMalloc((void**)&dev_idata, nNextHighestPowTwo * sizeof(int));
          checkCUDAError_SC("cudaMalloc idata failed");

          timer().startGpuTimer();
         
          cudaMemcpy((void*)dev_idata, (const void*)idata, nNextHighestPowTwo * sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAError_SC("cudaMemcpy idata failed");

          // call the kernel log2n number of times
          for (int i = 1; i <= ilog2ceil(nNextHighestPowTwo); ++i)
          {
            // call the kernel
            int twoToPowerIMinusOne = 1 << (i - 1);
            kernNaiveScan<<<((n + blockSize - 1) / blockSize) , blockSize>>>(nNextHighestPowTwo, twoToPowerIMinusOne, dev_gpuScanBuf, dev_idata);

            // flip flop the buffers 
            int* temp = dev_gpuScanBuf;
            dev_gpuScanBuf = dev_idata;
            dev_idata = temp;
          }

          // shift it and memcpy to out
          kernShiftScan << <((n + blockSize - 1) / blockSize), blockSize >> > (nNextHighestPowTwo, dev_gpuScanBuf, dev_idata);
        
          cudaMemcpy(odata, dev_gpuScanBuf, nNextHighestPowTwo * sizeof(float), cudaMemcpyDeviceToHost);

          timer().endGpuTimer();

          cudaFree(dev_gpuScanBuf);
          cudaFree(dev_idata);

        }
    }
}
