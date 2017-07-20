#include <sstream>
#include "beam.h"

using namespace std;

namespace amunmt {

void Beam::Add(const HypothesisPtr &hypo)
{
  assert(lineNum_ == hypo->GetLineNum());
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
const BeamPtr Beams::Get(size_t ind) const
{
  Coll::const_iterator iter = coll_.find(ind);
  if (iter == coll_.end()) {
    return BeamPtr();
  }
  else {
    return iter->second;
  }
}

BeamPtr Beams::at(size_t ind)
{
  Coll::const_iterator iter = coll_.find(ind);
  if (iter == coll_.end()) {
    BeamPtr beam(new Beam(ind));
    coll_[ind] = beam;
    return beam;
  }
  else {
    return iter->second;
  }
}

void Beams::Add(size_t ind, HypothesisPtr &hypo)
{
  assert(hypo);
  size_t lineNum = hypo->GetLineNum();
  BeamPtr beam = at(lineNum);
  assert(beam);
  beam->Add(hypo);
}


std::string Beams::Debug(size_t verbosity) const
{
  std::stringstream strm;

  strm << "size=" << size();

  if (verbosity) {
    for (const Coll::value_type &ele: coll_) {
      const Beam &beam = *ele.second;
      strm << endl << "\t" << beam.Debug(verbosity);
    }
  }

  return strm.str();
}


}

