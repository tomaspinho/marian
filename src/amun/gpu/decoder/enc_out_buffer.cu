#include "enc_out_buffer.h"

namespace amunmt {
namespace GPU {

EncOutBuffer::EncOutBuffer(unsigned int maxSize)
:buffer_(maxSize)
{
}

void EncOutBuffer::Add(EncOutPtr obj)
{
  buffer_.add(obj);
}

EncOutPtr EncOutBuffer::Get()
{
  return buffer_.remove();
}

}
}
