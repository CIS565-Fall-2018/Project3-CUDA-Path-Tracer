#pragma once

#include "common.h"


namespace StreamCompaction {
    namespace EfficientSM {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        int compact(int n, PathSegment *odata, const PathSegment *idata, int* indices_buff);
    }
}