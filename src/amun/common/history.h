#pragma once

#include <queue>
#include <algorithm>

#include "hypothesis.h"
#include "beam.h"

namespace amunmt {

class Sentences;
class God;

class History {
  private:
    struct HypothesisCoord {
      bool operator<(const HypothesisCoord& hc) const {
        return cost < hc.cost;
      }

      size_t i;
      size_t j;
      float cost;
    };

    History(const History&) = delete;

  public:
    History(const Sentence &sentence, bool normalizeScore, size_t maxLength);

    void Add(const Beam& beam);

    size_t size() const {
      return history_.size();
    }

    HypothesisPtr GetFirstHyps() const;

    NBestList NBest(size_t n) const;

    Result Top() const;

    size_t GetLineNum() const
    { return sentence_.GetLineNum(); }

    void Output(const God &god) const;

    void Output(const God &god, std::ostream& out) const;

  private:
    std::vector<Beam> history_;
    std::priority_queue<HypothesisCoord> topHyps_;
    bool normalize_;
    const Sentence &sentence_;
    size_t maxLength_;
};
///////////////////////////////////////////////////////////////////////////////////////

typedef std::shared_ptr<History> HistoryPtr;


} // namespace


