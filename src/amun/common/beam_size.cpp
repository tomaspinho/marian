#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize()
{
}

void BeamSize::Init(EncParamsPtr encParams)
{
  sizes_.resize(encParams->sentences->size(), 1);
  total_ = encParams->sentences->size();

  sentences_.resize(encParams->sentences->size());

  for (size_t i = 0; i < encParams->sentences->size(); ++i) {
    SentencePtr sentence = encParams->sentences->at(i);
    sentences_[i] = sentence;
  }
}

void BeamSize::Set(uint val)
{
  for (uint& beamSize : sizes_) {
    beamSize = val;
  }
  total_ = size() * val;
}

uint BeamSize::GetTotal() const
{
  return total_;
}

void BeamSize::Decr(size_t ind)
{
  --sizes_[ind];
  --total_;
}

std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << "sizes_=" << amunmt::Debug(sizes_, verbosity);
  strm << " sentences_=" << sentences_.size();

  return strm.str();
}

}


