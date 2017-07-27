#pragma once
#include "common/beam_size.h"
#include "../mblas/matrix.h"

namespace amunmt {
namespace GPU {

class BeamSizeGPU : public amunmt::BeamSize
{
public:
  BeamSizeGPU(mblas::EncParamsPtr encParams);

  mblas::IMatrix sentencesMask;

};


}
}
