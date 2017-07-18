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
    //coll_[i].reset(history);
    coll_[lineNum].reset(history);
    //std::cerr << "sentence=" << lineNum << " " << sentence.size() << std::endl;
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
      //HistoryPtr &history = coll_[i];

      size_t lineNum = beam.GetLineNum();
      for (size_t beamInd = 1; beamInd < beam.size(); ++beamInd) {
        assert(lineNum == beam.at(beamInd)->GetLineNum());
      }
      //std::cerr << "beam=" << beam.size() << " " << lineNum << std::endl;

      Coll::iterator iter = coll_.find(lineNum);
      assert(iter != coll_.end());
      HistoryPtr &history = iter->second;
      assert(history);

      history->Add(beam);

      if (beam.size() == 1 && beam.at(0)->GetWord() == EOS_ID) {
        std::cerr << "beam.size() == 1=" << lineNum << std::endl;
        history->Output(god);
        history.reset();
      }
    }
  }
}

Hypotheses Histories::GetFirstHyps() const
{
  Hypotheses hypos;

  for (size_t i = 0; i < sentences_.size(); ++i) {
    SentencePtr sentence = sentences_.at(i);
    size_t lineNum = sentence->GetLineNum();

    Coll::const_iterator iter = coll_.find(lineNum);
    assert(iter != coll_.end());
    const HistoryPtr &history = iter->second;

    HypothesisPtr hypo = history->front().at(0);
    hypos.push_back(hypo);
  }

  return hypos;
}

void Histories::OutputRemaining(const God &god) const
{
  for (const Coll::value_type &ele: coll_) {
    const HistoryPtr &history = ele.second;

    if (history) {
      history->Output(god);
    }
    else {
      cerr << "NONE\n";
    }
  }
}


} // namespace

