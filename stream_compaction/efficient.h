#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int npot, int *odata, const int *idata, int *cudaA);

        int compact(int n, int npot, int *odata, const int *idata, int *devCopy, int *devBinary);
    }
}
