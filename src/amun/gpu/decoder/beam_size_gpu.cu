#include "beam_size_gpu.h"
#include "gpu/mblas/matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

BeamSizeGPU::BeamSizeGPU()
:BeamSize()
,sentencesMask(nullptr)
,sourceContext(nullptr)
{
}

BeamSizeGPU::~BeamSizeGPU()
{}

void BeamSizeGPU::Init(EncParamsPtr encParams)
{
  BeamSize::Init(encParams);
  sentencesMask = &encParams->GetSentenceMask<mblas::CMatrix>();
  sourceContext = &encParams->GetSourceContext<mblas::Matrix>();
  sentenceLengths = &encParams->GetSentenceLengths<mblas::IMatrix>();
}

std::string BeamSizeGPU::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << amunmt::BeamSize::Debug(verbosity);
  strm << " sentencesMask=" << sentencesMask->Debug(0);
  strm << " sourceContext=" << sourceContext->Debug(0);

  return strm.str();
}

}
}
