#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize(SentencesPtr sentences)
:sizes_(sentences->size(), 1)
,sentences_(sentences->size())
,total_(sentences->size())
{
  for (size_t i = 0; i < sentences->size(); ++i) {
    SentencePtr sentence = sentences->at(i);
    sentences_[i] = sentence;
  }
}

void BeamSize::Init(uint val)
{
  for (uint& beamSize : sizes_) {
    beamSize = val;
  }
  total_ = size() * val;
}

void BeamSize::Decr(size_t ind)
{
  --sizes_[ind];
  --total_;
}

uint BeamSize::GetTotal() const
{
  return total_;
}


std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;
  strm << amunmt::Debug(sizes_, verbosity);
  return strm.str();
}

}


