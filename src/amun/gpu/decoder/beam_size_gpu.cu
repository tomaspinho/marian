#include "beam_size_gpu.h"

namespace amunmt {
namespace GPU {

BeamSizeGPU::BeamSizeGPU(mblas::EncParamsPtr encParams)
:BeamSize(encParams->sentences)
,sentencesMask(encParams->sentencesMask)
{

}


}
}

