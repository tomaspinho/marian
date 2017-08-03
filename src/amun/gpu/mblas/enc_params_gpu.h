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


protected:
  mblas::CMatrix sentencesMask_;
  mblas::Matrix sourceContext_;

  BaseMatrix &GetSentenceMaskInternal()
  { return sentencesMask_; }

  const BaseMatrix &GetSentenceMaskInternal() const
  { return sentencesMask_; }

  BaseMatrix &GetSourceContextInternal()
  { return sourceContext_; }

  const BaseMatrix &GetSourceContextInternal() const
  { return sourceContext_; }

};


/////////////////////////////////////////////////////////////////////////

}
}
}
