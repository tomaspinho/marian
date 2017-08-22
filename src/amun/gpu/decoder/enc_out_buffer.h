#pragma once
#include "enc_out_gpu.h"
#include "buffer.h"

namespace amunmt {
namespace GPU {

class EncOutBuffer
{
public:
  struct SentenceElement
  {
    EncOutPtr encOut;
    size_t sentenceInd; // index of the sentence we're translation within encOut.sentences

    SentenceElement(EncOutPtr vencOut,
                    size_t vsentenceInd)
    :encOut(vencOut)
    ,sentenceInd(vsentenceInd)
    {}

  };

  EncOutBuffer(unsigned int maxSize);

  void Add(EncOutPtr obj);
  EncOutPtr Get();

  void Get(size_t num, std::vector<SentenceElement> &ret);

protected:
  Buffer<EncOutPtr> buffer_;

  EncOutPtr unfinishedEncOutPtr_;
  size_t unfinishedInd_;
};


}
}
