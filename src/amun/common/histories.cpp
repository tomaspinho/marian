#include <iomanip>
#include "histories.h"
#include "sentences.h"
#include "god.h"

using namespace std;

namespace amunmt {

Histories::Histories(const Sentences& sentences, bool normalizeScore)
:sentences_(sentences)
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i).get();
    size_t lineNum = sentence.GetLineNum();
    History *history = new History(sentence, normalizeScore, 3 * sentence.size());
    coll_[i].reset(history);
    //coll_[lineNum].reset(history);
    std::cerr << "sentence=" << lineNum << " " << sentence.size() << std::endl;
  }
}

void Histories::AddAndOutput(const God &god, const Beams& beams)
{
  assert(size() == beams.size());

  for (size_t i = 0; i < size(); ++i) {
    const Beam &beam = beams.at(i);

    if (beam.empty()) {
      /*
      if (history) {
        history->Output(god);
        history.reset();
      }
      */
    }
    else {
      HistoryPtr &history = coll_[i];

      /*
      size_t lineNum = beam.at(0)->GetLineNum();
      for (size_t beamInd = 1; beamInd < beam.size(); ++beamInd) {
        assert(lineNum == beam.at(beamInd)->GetLineNum());
      }
      //std::cerr << "beam=" << beam.size() << " " << lineNum << std::endl;

      Coll::iterator iter = coll_.find(lineNum);
      assert(iter != coll_.end());
      HistoryPtr &history = iter->second;
      */
      assert(history);
      history->Add(beam);
    }
  }
}

Hypotheses Histories::GetFirstHyps() const
{
  cerr << "GetFirstHyps=";

  typedef std::map<size_t, HistoryPtr> OrderedMap;
  OrderedMap orderedMap;
  for (const Coll::value_type &ele: coll_) {
    orderedMap[ele.first] = ele.second;
  }

  Hypotheses hypos;

  for (const OrderedMap::value_type &ele: orderedMap) {
    const HistoryPtr &history = ele.second;
    HypothesisPtr hypo = history->front().at(0);
    cerr << ele.first << "=" << hypo->GetLineNum() << " ";
    hypos.push_back(hypo);
  }

  cerr << "sentences_=";
  for (size_t i = 0; i < sentences_.size(); ++i) {
    SentencePtr sentence = sentences_.at(i);
    cerr << sentence->GetLineNum() << " ";
  }

  cerr << endl;
  return hypos;
}

void Histories::OutputRemaining(const God &god) const
{
  for (const Coll::value_type &ele: coll_) {
    const HistoryPtr &history = ele.second;

    if (history) {
      history->Output(god);
    }
  }
}


} // namespace

