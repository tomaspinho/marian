#pragma once
#include "common/beam_size.h"
#include "../mblas/matrix.h"

namespace amunmt {
namespace GPU {

class BeamSizeGPU : public amunmt::BeamSize
{
public:
  mblas::CMatrix *sentencesMask;
  mblas::Matrix *sourceContext;

  BeamSizeGPU();
  ~BeamSizeGPU();

  void Init(EncParamsPtr encParams);

  std::string Debug(size_t verbosity = 1) const;

};


}
}
