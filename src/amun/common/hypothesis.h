#pragma once
#include <memory>
#include "common/types.h"
#include "common/soft_alignment.h"

namespace amunmt {

class Hypothesis;

typedef std::shared_ptr<Hypothesis> HypothesisPtr;

class Hypothesis {
  public:
    Hypothesis(size_t lineNum)
     : lineNum_(lineNum),
       prevHyp_(nullptr),
       prevIndex_(0),
       word_(0),
       cost_(0.0)
    {}

    Hypothesis(const HypothesisPtr prevHyp, size_t word, size_t prevIndex, float cost)
      : lineNum_(prevHyp->GetLineNum()),
        prevHyp_(prevHyp),
        prevIndex_(prevIndex),
        word_(word),
        cost_(cost)
    {}

    Hypothesis(const HypothesisPtr prevHyp, size_t word, size_t prevIndex, float cost,
               std::vector<SoftAlignmentPtr> alignment)
      : lineNum_(prevHyp->GetLineNum()),
        prevHyp_(prevHyp),
        prevIndex_(prevIndex),
        word_(word),
        cost_(cost),
        alignments_(alignment)
    {}

    size_t GetLineNum() const {
      return lineNum_;
    }

    const HypothesisPtr GetPrevHyp() const {
      return prevHyp_;
    }

    size_t GetWord() const {
      return word_;
    }

    size_t GetPrevStateIndex() const {
      return prevIndex_;
    }

    float GetCost() const {
      return cost_;
    }

    std::vector<float>& GetCostBreakdown() {
      return costBreakdown_;
    }

    SoftAlignmentPtr GetAlignment(size_t i) {
      return alignments_[i];
    }

    std::vector<SoftAlignmentPtr>& GetAlignments() {
      return alignments_;
    }

  private:
    size_t lineNum_;
    const HypothesisPtr prevHyp_;
    const size_t prevIndex_;
    const size_t word_;
    const float cost_;
    std::vector<SoftAlignmentPtr> alignments_;

    std::vector<float> costBreakdown_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////

class Beam
{
  typedef std::vector<HypothesisPtr> Coll;

public:
  typedef Coll::const_iterator const_iterator;

  const_iterator begin() const
  { return coll_.begin(); }

  const_iterator end() const
  { return coll_.end(); }

  Beam() {}

  Beam(std::initializer_list<HypothesisPtr> il)
  :coll_(il)
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
  Coll coll_;
};

//typedef std::vector<HypothesisPtr> Beam;

////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef std::vector<Beam> Beams;
typedef std::pair<Words, HypothesisPtr> Result;
typedef std::vector<Result> NBestList;

////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Debug(const Beam &vec, size_t verbosity = 1);
std::string Debug(const Beams &vec, size_t verbosity = 1);

}

