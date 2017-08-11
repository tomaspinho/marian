#pragma once
#include "enc_out_gpu.h"
#include "buffer.h"

namespace amunmt {
namespace GPU {

class EncOutBuffer
{
public:
  EncOutBuffer(unsigned int maxSize);

  void Add(EncOutPtr obj);
  EncOutPtr Get();

protected:
  Buffer<EncOutPtr> buffer_;

};


}
}
