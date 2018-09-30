#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          
          int sum = 0;
          for (int i = 0; i < n; ++i)
          {
            odata[i] = sum;
            sum += idata[i];
          }

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          
          int index = 0;
          for (int i = 0; i < n; ++i)
          {
            // if the data meets the condition put it in
            if (idata[i])
            {
              odata[index] = idata[i];
              ++index;
            }
          }

	        timer().endCpuTimer();
          return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
          int* scanned = (int*) malloc(sizeof(int) * n);

          timer().startCpuTimer();

          int sum = 0;
          for (int i = 0; i < n; ++i)
          {
            scanned[i] = sum;
            if (idata[i]) 
            {
              ++sum;
            }
          }

          // now scatter
          for (int j = 0; j < n; j++)
          {
            if (idata[j])
            {
              odata[scanned[j]] = idata[j];
            }
          }

	        timer().endCpuTimer();

          free(scanned);
          return sum;
        }
    }
}
