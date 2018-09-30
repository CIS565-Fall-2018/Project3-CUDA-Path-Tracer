set(SOURCE_FILES
	"tiny_obj_loader.cc"
    "tiny_obj_loader.h"
    )

cuda_add_library(tinyobj
    ${SOURCE_FILES}
    )