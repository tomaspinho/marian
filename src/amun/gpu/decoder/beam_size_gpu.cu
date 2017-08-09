#include "beam_size_gpu.h"
#include "gpu/mblas/matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

BeamSizeGPU::BeamSizeGPU()
:BeamSize()
{
}

BeamSizeGPU::~BeamSizeGPU()
{}

void BeamSizeGPU::Init(uint maxBeamSize, EncOutPtr encOut)
{
  BeamSize::Init(maxBeamSize, encOut);
  sourceContext_.reset(new mblas::Matrix(encOut->GetSourceContext<mblas::Matrix>()));
  sentenceLengths_.reset(new mblas::IMatrix(encOut->GetSentenceLengths<mblas::IMatrix>()));
}

std::string BeamSizeGPU::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << amunmt::BeamSize::Debug(verbosity);
  strm << " sourceContext_=" << sourceContext_->Debug(0);
  strm << " sentenceLengths_=" << sentenceLengths_->Debug(0);

  return strm.str();
}

}
}
