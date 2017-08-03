#pragma once
#include "common/beam_size.h"
#include "gpu/mblas/matrix.h"

namespace amunmt {
namespace GPU {

class BeamSizeGPU : public amunmt::BeamSize
{
public:
  mblas::Matrix *sourceContext;
  const mblas::IMatrix *sentenceLengths;

  BeamSizeGPU();
  ~BeamSizeGPU();

  void Init(EncParamsPtr encParams);

  std::string Debug(size_t verbosity = 1) const;

};


}
}
