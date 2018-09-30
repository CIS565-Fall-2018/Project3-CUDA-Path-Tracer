
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

int* dev_efficientScanBuf;
int* dev_efficientBools;
int* dev_efficientIndices;
PathSegment** dev_odata_buffer;

__global__ void kernEfficientScanUpSweep(int n, int d, int* odata, int* idata)
{
  // get index first
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  int twoToPowDPlusOne = 1 << (d + 1);
  if (index >= n || index % twoToPowDPlusOne != 0)
  {
    return;
  }
  
  int twoToPowD = 1 << d;

  // then add the two numbers and put them into the global output buffer
  odata[index + twoToPowDPlusOne - 1] = idata[index + twoToPowDPlusOne - 1] + idata[index + twoToPowD - 1];
}

__global__ void kernSetFirstElementZero(int n, int* odata)
{
  odata[n - 1] = 0;
}

__global__ void kernEfficientScanDownSweep(int n, int d, int* odata, int* idata)
{
  // get index first
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  int twoToPowDPlusOne = 1 << (d + 1);
  if (index >= n || (index % twoToPowDPlusOne != 0))
  {
    return;
  }
  
  int twoToPowD = 1 << d;

  // then sweep down
  odata[index + twoToPowD - 1] = idata[index + twoToPowDPlusOne - 1];
  odata[index + twoToPowDPlusOne - 1] = idata[index + twoToPowDPlusOne - 1] + idata[index + twoToPowD - 1];
}

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, PathSegment **dev_odata, PathSegment **dev_idata) {
            int nNextHighestPowTwo = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_efficientBools, nNextHighestPowTwo * sizeof(int));
            checkCUDAError_SC("cudaMalloc bool buf failed");

            cudaMalloc((void**)&dev_efficientScanBuf, nNextHighestPowTwo * sizeof(int));
            checkCUDAError_SC("cudaMalloc buf failed");

            cudaMalloc((void**)&dev_efficientIndices, nNextHighestPowTwo * sizeof(int));
            checkCUDAError_SC("cudaMalloc indices failed");
            
            cudaMalloc((void***)&dev_odata_buffer, nNextHighestPowTwo * sizeof(PathSegment*));
            checkCUDAError_SC("cudaMalloc indices failed");

            // map all of the values to booleans (and pad with zeroes for those values higher than original array limit)
            StreamCompaction::Common::kernMapToBoolean<< <((nNextHighestPowTwo + blockSize - 1) / blockSize), blockSize >> > (n, nNextHighestPowTwo, dev_efficientBools, dev_idata);
            checkCUDAError_SC("kern map to boolean");

            // Start the scan --------------- (copy pasted from the scan function because you can't nest calls to timer. Plus it saves a copy from device to host)

            // make a copy of the bools so we can do the scan and put it into indices
            cudaMemcpy((void*)dev_efficientIndices, (const void*)dev_efficientBools, nNextHighestPowTwo * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError_SC("cudaMemcpy idata failed");

            // call the upsweep kernel log2n number of times
            for (int d = 0; d < ilog2ceil(nNextHighestPowTwo); ++d)
            {
              // copy all the data to make sure everythings in place
              cudaMemcpy((void*)dev_efficientScanBuf, (const void*)dev_efficientIndices, nNextHighestPowTwo * sizeof(int), cudaMemcpyDeviceToDevice);
              checkCUDAError_SC("cudaMemcpy idata failed");
              
              // call the kernel
              kernEfficientScanUpSweep << <((nNextHighestPowTwo + blockSize - 1) / blockSize), blockSize >> > (nNextHighestPowTwo, d, dev_efficientScanBuf, dev_efficientIndices);
              checkCUDAError_SC("Scan up sweep");

              // flip flop the buffers so that idata is always the most recent data
              int* temp = dev_efficientScanBuf;
              dev_efficientScanBuf = dev_efficientIndices;
              dev_efficientIndices = temp;
            }
            
            // set first element to be zero in a new kernel (unsure how to do this otherwise)
            kernSetFirstElementZero << <1, 1 >> > (nNextHighestPowTwo, dev_efficientIndices);
            checkCUDAError_SC("set first element zero failed");

            // now call the downsweep kernel log2n times
            for (int d = (ilog2ceil(nNextHighestPowTwo) - 1); d >= 0; --d)
            {
              // copy all the data to make sure everythings in place
              cudaMemcpy((void*)dev_efficientScanBuf, (const void*)dev_efficientIndices, nNextHighestPowTwo * sizeof(int), cudaMemcpyDeviceToDevice);
              checkCUDAError_SC("cudaMemcpy idata failed");
              
              // call the kernel
              kernEfficientScanDownSweep << <((nNextHighestPowTwo + blockSize - 1) / blockSize), blockSize >> > (nNextHighestPowTwo, d, dev_efficientScanBuf, dev_efficientIndices);
              checkCUDAError_SC("Scan downsweep");

              // flip flop the buffers
              int* temp = dev_efficientScanBuf;
              dev_efficientScanBuf = dev_efficientIndices;
              dev_efficientIndices = temp;
            }

            // ------- end of scan

            int sizeOfCompactedStream = 0;
            // memcpy the final value of indices to out so that we can get the total size of compacted stream
            cudaMemcpy(&sizeOfCompactedStream, dev_efficientIndices + (nNextHighestPowTwo - 1), 1 * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError_SC("memcpy failed");

            // run the stream compaction
            StreamCompaction::Common::kernScatter << <((nNextHighestPowTwo + blockSize - 1) / blockSize), blockSize >> > (n, dev_odata_buffer, dev_idata, dev_efficientBools, dev_efficientIndices);
            checkCUDAError_SC("Scatter failed");

            cudaMemcpy(dev_odata, dev_odata_buffer, sizeOfCompactedStream * sizeof(PathSegment*), cudaMemcpyDeviceToDevice);

            // free all our stuff
            cudaFree(dev_efficientScanBuf);
            cudaFree(dev_efficientBools);
            cudaFree(dev_efficientIndices);
            cudaFree(dev_odata_buffer);

            // return the total size of the compacted stream
            return sizeOfCompactedStream;
        }
    }
}
