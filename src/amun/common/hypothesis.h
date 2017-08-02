#pragma once
#include <memory>
#include "common/types.h"
#include "common/soft_alignment.h"
#include "common/sentence.h"

namespace amunmt {

class Hypothesis;

typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef std::vector<HypothesisPtr> Hypotheses;

class Hypothesis {
  public:
    Hypothesis(const Sentence &sentence)
     : sentence_(sentence),
       prevHyp_(nullptr),
       prevIndex_(0),
       word_(0),
       cost_(0.0),
       numWords_(0)
    {}

    Hypothesis(const HypothesisPtr prevHyp, size_t word, size_t prevIndex, float cost)
      : sentence_(prevHyp->sentence_),
        prevHyp_(prevHyp),
        prevIndex_(prevIndex),
        word_(word),
        cost_(cost),
        numWords_(prevHyp->numWords_ + 1)
    {}

    Hypothesis(const HypothesisPtr prevHyp, size_t word, size_t prevIndex, float cost,
               std::vector<SoftAlignmentPtr> alignment)
      : sentence_(prevHyp->sentence_),
        prevHyp_(prevHyp),
        prevIndex_(prevIndex),
        word_(word),
        cost_(cost),
        alignments_(alignment),
        numWords_(prevHyp->numWords_ + 1)
    {}

    size_t GetLineNum() const {
      return sentence_.GetLineNum();
    }

    const HypothesisPtr GetPrevHyp() const {
      return prevHyp_;
    }

    size_t GetWord() const {
      return word_;
    }

    size_t GetNumWords() const {
      return numWords_;
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

    const Sentence &GetSentence() const {
      return sentence_;
    }

  private:
    const HypothesisPtr prevHyp_;
    const Sentence &sentence_;
    const size_t prevIndex_;
    const size_t word_;
    const float cost_;
    const size_t numWords_;
    std::vector<SoftAlignmentPtr> alignments_;

    std::vector<float> costBreakdown_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef std::pair<Words, HypothesisPtr> Result;
typedef std::vector<Result> NBestList;

}

