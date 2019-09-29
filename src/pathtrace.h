#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, 
	int frame, 
	int iteration, 
	bool enable_stream_compact = false, 
	int enable_material_batching = 0, 
	bool enable_cache_first_path = false,
	int integrator_type = 0);//0-niave, 1-direct-light, 2-direct-light-mis, 3-full light
