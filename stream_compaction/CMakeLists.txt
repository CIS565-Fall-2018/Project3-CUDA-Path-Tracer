set(SOURCE_FILES
	"stream_compact.h"
	"stream_compact.cu"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
