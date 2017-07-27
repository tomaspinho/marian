#pragma once
#include "common/beam_size.h"

namespace amunmt {
namespace GPU {

class BeamSizeGPU : public amunmt::BeamSize
{
public:
  BeamSizeGPU(SentencesPtr sentences);

};


}
}
