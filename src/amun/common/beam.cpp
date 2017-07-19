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
Beams::Beams(SentencesPtr sentences)
:coll_(sentences->size())
{
  size_t size = sentences->size();
  for (size_t i = 0; i < size; ++i) {
    Beam *beam = new Beam(999);
    coll_[i].reset(beam);
  }
}

const BeamPtr Beams::Get(size_t ind) const
{
  return coll_.at(ind);
}

Beam &Beams::at(size_t ind)
{
  return *coll_.at(ind);
}

void Beams::Add(size_t ind, HypothesisPtr &hypo)
{
  Beam &beam = at(ind);
  beam.Add(hypo);
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

