#include "beam_size_gpu.h"
#include "gpu/mblas/matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

BeamSizeGPU::BeamSizeGPU()
:BeamSize()
,sourceContext_(nullptr)
{
}

BeamSizeGPU::~BeamSizeGPU()
{}

void BeamSizeGPU::Init(EncParamsPtr encParams)
{
  BeamSize::Init(encParams);
  sourceContext_ = &encParams->GetSourceContext<mblas::Matrix>();
  sentenceLengths_ = &encParams->GetSentenceLengths<mblas::IMatrix>();
}

std::string BeamSizeGPU::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << amunmt::BeamSize::Debug(verbosity);
  strm << " sourceContext_=" << sourceContext_->Debug(0);

  return strm.str();
}

}
}

