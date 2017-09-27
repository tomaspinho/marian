#pragma once
#include <cuda_fp16.h>

#define __fp16 half

__fp16 float2half_rn (float a);
