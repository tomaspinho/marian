#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize(SentencesPtr sentences)
:std::vector<uint>(sentences->size(), 1)
,total_(sentences->size())
{

}

void BeamSize::Init(uint val)
{
  for (uint& beamSize : *this) {
    beamSize = val;
  }
  total_ = size() * val;
}

/*


std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;
  strm << amunmt::Debug(vec_, verbosity);
}
*/
}


