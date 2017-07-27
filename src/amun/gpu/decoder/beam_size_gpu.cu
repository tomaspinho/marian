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

void BeamSizeGPU::DeleteEmpty()
{
  size_t i = 0;
  while (i < size()) {
    if (sizes_[i]) {
      ++i;
    }
    else {
      sizes_.erase(sizes_.begin() + i);
      sentences_.erase(sentences_.begin() + i);
    }
  }
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

