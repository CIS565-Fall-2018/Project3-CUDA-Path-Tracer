#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        int compact(int n, PathSegment **odata, PathSegment **idata);
    }
}
