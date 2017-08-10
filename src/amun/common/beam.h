#pragma once
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include "hypothesis.h"
#include "sentences.h"

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

  size_t GetLineNum() const
  { return lineNum_; }

  size_t size() const
  { return coll_.size(); }

  const HypothesisPtr &at(size_t ind) const
  { return coll_.at(ind); }

  const HypothesisPtr &back() const
  { return coll_.back(); }

  void Add(const HypothesisPtr &hypo);

  void swap (Beam &other)
  {
    coll_.swap(other.coll_);
  }

  std::string Debug(size_t verbosity = 1) const;

protected:
  size_t lineNum_;
  Coll coll_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef std::shared_ptr<Beam> BeamPtr;

////////////////////////////////////////////////////////////////////////////////////////////////////////

class Beams
{
public:
  //typedef std::vector<BeamPtr> Coll;
  typedef std::unordered_map<size_t, BeamPtr> Coll;
  typedef Coll::const_iterator const_iterator;

  const_iterator begin() const
  { return coll_.begin(); }

  const_iterator end() const
  { return coll_.end(); }

  size_t size() const
  { return coll_.size(); }

  std::pair<bool, const Beam*> Get(size_t lineNum) const;

  void Add(size_t ind, HypothesisPtr &hypo);

  std::string Debug(size_t verbosity = 1) const;

protected:
  Coll coll_;

  BeamPtr Get(size_t lineNum);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////


}

