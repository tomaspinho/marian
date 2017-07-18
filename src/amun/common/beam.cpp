#include <sstream>
#include "beam.h"

using namespace std;

namespace amunmt {

void Beam::Add(const HypothesisPtr &hypo)
{
  if (coll_.empty()) {
    lineNum_ = hypo->GetLineNum();
  }
  else {
    assert(lineNum_ == hypo->GetLineNum());
  }
  coll_.push_back(hypo);
}

std::string Beam::Debug(size_t verbosity) const
{
  std::stringstream strm;

  strm << "size=" << size();

  if (verbosity) {
    for (size_t i = 0; i < size(); ++i) {
      const HypothesisPtr &hypo = coll_[i];
      strm << " " << hypo->GetWord();
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

std::string Beams::Debug(size_t verbosity) const
{
  std::stringstream strm;

  strm << "size=" << size();

  if (verbosity) {
    for (size_t i = 0; i < size(); ++i) {
      const Beam &beam = *coll_[i];
      strm << endl << "\t" << beam.Debug(verbosity);
    }
  }

  return strm.str();
}


}

