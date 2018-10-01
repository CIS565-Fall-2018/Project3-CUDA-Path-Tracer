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
	        //timer().startCpuTimer();
            // TODO	
			odata[0] = 0;
			for (int i = 1; i < n; ++i) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
	        //timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
			int index = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[index++] = idata[i];
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
			// allocate space for two middle status arrays
			int *temp1 = (int*)malloc(n * sizeof(int));
			int *temp2 = (int*)malloc(n * sizeof(int));
			timer().startCpuTimer();
	        // TODO
			for (int i = 0; i < n; ++i) {
				temp1[i] = idata[i] == 0 ? 0 : 1;
			}
			scan(n, temp2, temp1);
			int index = 0;
			for (int i = 0; i < n; ++i) {
				if (temp1[i] == 1) {
					odata[temp2[i]] = idata[i];
					index++;
				}
			}
	        timer().endCpuTimer();
            return index;
        }
    }
}
