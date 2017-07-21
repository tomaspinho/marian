#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {
BeamSize::BeamSize(size_t size, uint val)
:std::vector<uint>(size, val)
{

}

/*
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
  total_ = vec_.size() * val;
}

std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;
  strm << amunmt::Debug(vec_, verbosity);
}
*/
}


