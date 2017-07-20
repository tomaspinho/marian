#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize(SentencesPtr sentences)
:vec_(sentences->size(), 1)
{

}

void BeamSize::Init(uint val)
{
  for (uint& beamSize : vec_) {
    beamSize = val;
  }
}

std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;
  strm << amunmt::Debug(vec_, verbosity);
}

}


