#pragma once
#include "common/beam_size.h"
#include "../mblas/matrix.h"

namespace amunmt {
namespace GPU {

class BeamSizeGPU : public amunmt::BeamSize
{
public:
  mblas::IMatrix sentencesMask;
  mblas::Matrix sourceContext;

  BeamSizeGPU(mblas::EncParamsPtr encParams);

  void DeleteEmpty();

  std::string Debug(size_t verbosity = 1) const;

};


}
}
