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

		void printArr(int n, int *arr) {
			printf("[");
			for (int i = 0; i < n; i++) {
				printf("%d ", arr[i]);
			}
			printf("]");
		}

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
			bool end = true;
			try {
				timer().startCpuTimer();
			}
			catch (std::exception) {
				end = false;
			}
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = idata[i - 1] + odata[i - 1];
			}
			if (end) {
				timer().endCpuTimer();
			}
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int odataIdx = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[odataIdx] = idata[i];
					odataIdx++;
				}
			}
	        timer().endCpuTimer();
            return odataIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int *tmpArr = new int[n];
			int *scanRes = new int[n];
			for (int i = 0; i < n; i++) {
				tmpArr[i] = idata[i] != 0 ? 1 : 0;
			}

			scan(n, scanRes, tmpArr);
			//printf("scanRes: ");
			//printArr(n, scanRes);

			for (int i = 0; i < n; i++) {
				if (tmpArr[i] == 1) {
					odata[scanRes[i]] = idata[i];
				}
			}
			
	        timer().endCpuTimer();
			return scanRes[n - 1];
        }
    }
}
