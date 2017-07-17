#pragma once
#include <vector>
#include <string>
#include "hypothesis.h"

namespace amunmt {

class Beam
{
  typedef std::vector<HypothesisPtr> Coll;

public:
  typedef Coll::const_iterator const_iterator;

  const_iterator begin() const
  { return coll_.begin(); }

  const_iterator end() const
  { return coll_.end(); }

  Beam(size_t lineNum)
  :lineNum_(lineNum)
  {}

  Beam(size_t lineNum, std::initializer_list<HypothesisPtr> il)
  :lineNum_(lineNum)
  ,coll_(il)
  {}

  size_t size() const
  { return coll_.size(); }

  const HypothesisPtr &at(size_t ind) const
  { return coll_.at(ind); }

  HypothesisPtr &at(size_t ind)
  { return coll_.at(ind); }

  const HypothesisPtr &back() const
  { return coll_.back(); }

  bool empty() const
  { return coll_.empty(); }

  void push_back(const HypothesisPtr &hypo)
  {
    coll_.push_back(hypo);
  }

  void swap (Beam &other)
  {
    coll_.swap(other.coll_);
  }

protected:
  size_t lineNum_;
  Coll coll_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////

class Beams
{
  typedef std::shared_ptr<Beam> BeamPtr;
  typedef std::vector<BeamPtr> Coll;
public:
  Beams(size_t size);

  size_t size() const
  { return coll_.size(); }

  const Beam &at(size_t ind) const
  { return *coll_.at(ind); }

  Beam &at(size_t ind)
  { return *coll_.at(ind); }

protected:
  Coll coll_;
};

//typedef std::vector<Beam> Beams;

////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Debug(const Beam &vec, size_t verbosity = 1);
std::string Debug(const Beams &vec, size_t verbosity = 1);

}

