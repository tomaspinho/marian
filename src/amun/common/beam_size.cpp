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

uint BeamSize::GetTotal() const
{
  return total_;
}

void BeamSize::Decr(size_t ind)
{
  --sizes_[ind];
  --total_;
}

void BeamSize::DeleteEmpty()
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


std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;
  strm << amunmt::Debug(sizes_, verbosity);
  return strm.str();
}

}


