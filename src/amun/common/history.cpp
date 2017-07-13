#include "history.h"
#include "sentences.h"

namespace amunmt {

History::History(size_t lineNo, bool normalizeScore, size_t maxLength)
  : normalize_(normalizeScore),
    lineNo_(lineNo),
   maxLength_(maxLength)
{
  Add({HypothesisPtr(new Hypothesis())});
}

void History::Add(const Beam& beam) {
  if (beam.back()->GetPrevHyp() != nullptr) {
    for (size_t j = 0; j < beam.size(); ++j)
      if(beam[j]->GetWord() == EOS_ID || size() == maxLength_ ) {
        float cost = normalize_ ? beam[j]->GetCost() / history_.size() : beam[j]->GetCost();
        topHyps_.push({ history_.size(), j, cost });
      }
  }
  history_.push_back(beam);
}

NBestList History::NBest(size_t n) const {
  NBestList nbest;
  auto topHypsCopy = topHyps_;
  while (nbest.size() < n && !topHypsCopy.empty()) {
    auto bestHypCoord = topHypsCopy.top();
    topHypsCopy.pop();

    size_t start = bestHypCoord.i;
    size_t j  = bestHypCoord.j;

    Words targetWords;
    HypothesisPtr bestHyp = history_[start][j];
    while(bestHyp->GetPrevHyp() != nullptr) {
      targetWords.push_back(bestHyp->GetWord());
      bestHyp = bestHyp->GetPrevHyp();
    }

    std::reverse(targetWords.begin(), targetWords.end());
    nbest.emplace_back(targetWords, history_[bestHypCoord.i][bestHypCoord.j]);
  }
  return nbest;
}


/////////////////////////////////////////////////////////////////////////////////////////////////

Histories::Histories(const Sentences& sentences, bool normalizeScore)
 : coll_(sentences.size())
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i).get();
    History *history = new History(sentence.GetLineNum(), normalizeScore, 3 * sentence.size());
    coll_[i].reset(history);
  }
}


class LineNumOrderer
{
  public:
    bool operator()(const HistoryPtr& a, const HistoryPtr& b) const
    {
      return a->GetLineNum() < b->GetLineNum();
    }
};


void Histories::SortByLineNum()
{
  std::sort(coll_.begin(), coll_.end(), LineNumOrderer());
}


void Histories::Append(const Histories &other)
{
  for (size_t i = 0; i < other.size(); ++i) {
    HistoryPtr history = other.coll_[i];
    coll_.push_back(history);
  }
}

}

