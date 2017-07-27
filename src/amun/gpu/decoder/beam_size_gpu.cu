#include "beam_size_gpu.h"

using namespace std;

namespace amunmt {
namespace GPU {

BeamSizeGPU::BeamSizeGPU(mblas::EncParamsPtr encParams)
:BeamSize(encParams->sentences)
,sentencesMask(encParams->GetSentenceMask())
,sourceContext(encParams->GetSourceContext())
{

}

std::string BeamSizeGPU::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << amunmt::BeamSize::Debug(verbosity);
  strm << " sentencesMask=" << sentencesMask.Debug(0);
  strm << " sourceContext=" << sourceContext.Debug(0);

  return strm.str();
}

}
}

