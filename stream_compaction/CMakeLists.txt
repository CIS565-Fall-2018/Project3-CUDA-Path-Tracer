set(SOURCE_FILES
        "common.cu"
        "common.h"
        "efficient_sm.cu"
        "efficient_sm.h"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
