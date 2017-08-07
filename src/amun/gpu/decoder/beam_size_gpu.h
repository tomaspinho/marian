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

  void Init(uint maxBeamSize, EncOutPtr encOut);

  const mblas::Matrix &GetSourceContext() const
  { return *sourceContext_; }

  const mblas::IMatrix &GetSentenceLengths() const
  { return *sentenceLengths_; }

  std::string Debug(size_t verbosity = 1) const;

protected:
  std::unique_ptr<mblas::Matrix> sourceContext_;
  std::unique_ptr<mblas::IMatrix> sentenceLengths_;

};


}
}
