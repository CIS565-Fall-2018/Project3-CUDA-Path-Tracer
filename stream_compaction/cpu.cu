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
			for (int i = 1; i < n; i++) {
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
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0) {
					continue;
				}
				odata[count++] = idata[i];
			}
	        timer().endCpuTimer();
            return count;
        }

        /**s
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
			int *mdata = new int[n];
			int *sdata = new int[n];
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0) {
					mdata[i] = 0;
				}
				else {
					mdata[i] = 1;
				}
			}
			scan(n, sdata, mdata);
			for (int i = 0; i < n; i++) {
				if (mdata[i] != 0) {
					odata[sdata[i]] = idata[i];
					count++;
				}
			}
			delete[] mdata, sdata;
	        timer().endCpuTimer();
            return count;
        }
    }
}
