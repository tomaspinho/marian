#pragma once

#include "../mblas/matrix.h"
#include "common/enc_params.h"

namespace amunmt {
namespace GPU {
namespace mblas {

class EncParamsGPU : public EncParams
{
public:
  EncParamsGPU(SentencesPtr sentences);


protected:
  mblas::Matrix sourceContext_;
  mblas::IMatrix sentenceLengths_;

  BaseMatrix &GetSourceContextInternal()
  { return sourceContext_; }

  const BaseMatrix &GetSourceContextInternal() const
  { return sourceContext_; }

  virtual const BaseMatrix &GetSentenceLengthsInternal() const
  { return sentenceLengths_; }

};


/////////////////////////////////////////////////////////////////////////

}
}
}
