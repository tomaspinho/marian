#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize(SentencesPtr sentences)
{
  Init(sentences);
}

void BeamSize::Init(SentencesPtr sentences)
{
  sizes_.resize(sentences->size(), 1);
  sentences_.resize(sentences->size());
  total_ = sentences->size();

  for (size_t i = 0; i < sentences->size(); ++i) {
    SentencePtr sentence = sentences->at(i);
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


