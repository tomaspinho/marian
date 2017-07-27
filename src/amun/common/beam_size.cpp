#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize(SentencesPtr sentences)
:vec_(sentences->size(), 1)
,total_(sentences->size())
{

}

void BeamSize::Init(uint val)
{
  for (uint& beamSize : vec_) {
    beamSize = val;
  }
  total_ = size() * val;
}

void BeamSize::Decr(size_t ind)
{
  --vec_[ind];
  --total_;
}

uint BeamSize::GetTotal() const
{
  return total_;
}


std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;
  strm << amunmt::Debug(vec_, verbosity);
  return strm.str();
}

}


