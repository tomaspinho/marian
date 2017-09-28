#pragma once
#include <cuda_fp16.h>
#include <iostream>

#define __fp16 half

std::ostream& operator<<(std::ostream& os, const half &val);

__fp16 float2half_rn (float a);

