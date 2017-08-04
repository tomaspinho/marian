#pragma once

#include "../mblas/matrix.h"
#include "common/enc_out.h"

namespace amunmt {
namespace GPU {
namespace mblas {

class EncOutGPU : public EncOut
{
public:
  EncOutGPU(SentencesPtr sentences);


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
