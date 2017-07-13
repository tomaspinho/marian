#pragma once

#include <queue>
#include <algorithm>

#include "hypothesis.h"

namespace amunmt {

class Sentences;

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
    History(size_t lineNo, bool normalizeScore, size_t maxLength);

    void Add(const Beam& beam);

    size_t size() const {
      return history_.size();
    }

    Beam& front() {
      return history_.front();
    }

    NBestList NBest(size_t n) const;

    Result Top() const {
      return NBest(1)[0];
    }

    size_t GetLineNum() const
    { return lineNo_; }

  private:
    std::vector<Beam> history_;
    std::priority_queue<HypothesisCoord> topHyps_;
    bool normalize_;
    size_t lineNo_;
    size_t maxLength_;
};
///////////////////////////////////////////////////////////////////////////////////////

typedef std::shared_ptr<History> HistoryPtr;

///////////////////////////////////////////////////////////////////////////////////////

class Histories {
  public:
    Histories() {} // for all histories in translation task
    Histories(const Sentences& sentences, bool normalizeScore);

    HistoryPtr at(size_t id) const
    { return coll_.at(id); }

    size_t size() const {
      return coll_.size();
    }

    void Add(const Beams& beams) {
      for (size_t i = 0; i < size(); ++i) {
        if (!beams[i].empty()) {
          coll_[i]->Add(beams[i]);
        }
      }
    }

    void SortByLineNum();
    void Append(const Histories &other);

    Beam GetFirstHyps() {
      Beam beam;
      for (auto& history : coll_) {
        beam.emplace_back(history->front()[0]);
      }
      return beam;
    }

  protected:
    std::vector<HistoryPtr> coll_;
    Histories(const Histories &) = delete;
};

///////////////////////////////////////////////////////////////////////////////////////

typedef std::shared_ptr<Histories> HistoriesPtr;

} // namespace


