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
			odata[0] = 0;
			for (int k = 1; k < n; ++k) {
				odata[k] = idata[k - 1] + odata[k - 1];
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
            // TODO
			int counter = 0;
			for (int k = 0; k < n; ++k) {
				if (idata[k] != 0) {
					odata[counter] = idata[k];
					counter++;
				}
			}
	        timer().endCpuTimer();
			return counter;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {

			int *binaryArray = new int[n];
			int *scanResult = new int[n];

			timer().startCpuTimer();
			// prepare temporary binary array
			for (int k = 0; k < n; ++k) {
				if (idata[k] != 0) {
					binaryArray[k] = 1;
				}
				else {
					binaryArray[k] = 0;
				}
			}
			// scan
			scanResult[0] = 0;
			for (int k = 1; k < n; ++k) {
				scanResult[k] = binaryArray[k - 1] + scanResult[k - 1];
			}
			// scatter
			int counter = 0;
			for (int k = 0; k < n; ++k) {
				if (binaryArray[k] == 1) {
					odata[scanResult[k]] = idata[k];
					counter++;
				}
			}
			timer().endCpuTimer();

			delete binaryArray;
			delete scanResult;

			return counter;
        }
    }
}
