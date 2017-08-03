#include "beam_size_gpu.h"
#include "gpu/mblas/matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

BeamSizeGPU::BeamSizeGPU()
:BeamSize()
,sourceContext(nullptr)
{
}

BeamSizeGPU::~BeamSizeGPU()
{}

void BeamSizeGPU::Init(EncParamsPtr encParams)
{
  BeamSize::Init(encParams);
  sourceContext = &encParams->GetSourceContext<mblas::Matrix>();
  sentenceLengths = &encParams->GetSentenceLengths<mblas::IMatrix>();
}

std::string BeamSizeGPU::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << amunmt::BeamSize::Debug(verbosity);
  strm << " sourceContext=" << sourceContext->Debug(0);

  return strm.str();
}

}
}

