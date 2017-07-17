#include <sstream>
#include "beam.h"

using namespace std;

namespace amunmt {

std::string Debug(const Beam &vec, size_t verbosity)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    for (size_t i = 0; i < vec.size(); ++i) {
      const HypothesisPtr &hypo = vec.at(i);
      strm << " " << hypo->GetWord();
    }
  }

  return strm.str();
}

std::string Debug(const Beams &vec, size_t verbosity)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    for (size_t i = 0; i < vec.size(); ++i) {
      const Beam &beam = vec.at(i);
      strm << endl << "\t" << Debug(beam, verbosity);
    }
  }

  return strm.str();
}

//////////////////////////////////////////////////////////////////////////////////////////////
Beams::Beams(size_t size)
:coll_(size)
{
  for (size_t i = 0; i < size; ++i) {
    Beam *beam = new Beam(999);
    coll_[i].reset(beam);
  }
}


}

