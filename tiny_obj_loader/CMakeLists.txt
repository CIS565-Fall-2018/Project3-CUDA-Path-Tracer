set(SOURCE_FILES
    "tiny_obj_loader.cc"
    "tiny_obj_loader.h"
    "loader_example.cc"
    )

cuda_add_library(tiny_obj_loader
    ${SOURCE_FILES}
    OPTIONS -arch=sm_50
    )
