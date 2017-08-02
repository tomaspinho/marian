#pragma once

#include "matrix.h"
#include "common/enc_params.h"

namespace amunmt {
namespace GPU {
namespace mblas {

class EncParamsGPU : public EncParams
{
public:
  EncParamsGPU() {}

  virtual void SetSentences(const SentencesPtr sentences);

  BaseMatrix &GetSentenceMask()
  { return sentencesMask_; }

  const BaseMatrix &GetSentenceMask() const
  { return sentencesMask_; }

  BaseMatrix &GetSourceContext()
  { return sourceContext_; }

  const BaseMatrix &GetSourceContext() const
  { return sourceContext_; }

protected:
  mblas::CMatrix sentencesMask_;
  mblas::Matrix sourceContext_;

};


/////////////////////////////////////////////////////////////////////////

}
}
}
