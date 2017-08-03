#pragma once
#include "common/beam_size.h"
#include "gpu/mblas/matrix.h"

namespace amunmt {
namespace GPU {

class BeamSizeGPU : public amunmt::BeamSize
{
public:
  BeamSizeGPU();
  ~BeamSizeGPU();

  void Init(EncParamsPtr encParams);

  const mblas::Matrix &GetSourceContext() const
  { return *sourceContext_; }

  const mblas::IMatrix &GetSentenceLengths() const
  { return *sentenceLengths_; }

  std::string Debug(size_t verbosity = 1) const;

protected:
  mblas::Matrix *sourceContext_;
  const mblas::IMatrix *sentenceLengths_;

};


}
}
