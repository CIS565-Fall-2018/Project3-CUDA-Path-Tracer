#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
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
            
			thrust::host_vector<int> hv(n);
			thrust::copy(idata, idata + n, hv.begin());
			thrust::device_vector<int> i_dv = hv;
			thrust::device_vector<int> o_dv(n);

			timer().startGpuTimer();
			thrust::exclusive_scan(i_dv.begin(), i_dv.end(), o_dv.begin());
            timer().endGpuTimer();

			thrust::copy(o_dv.begin(), o_dv.end(), odata);
        }
    }
}
