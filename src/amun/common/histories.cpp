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
  assert(size() <= beams.size());

  for (const Beams::Coll::value_type &ele: beams) {
    const Beam &beam = *ele.second;
    assert(!beam.empty());

    size_t lineNum = beam.GetLineNum();
    //std::cerr << "beam=" << lineNum << " " << beam.size() << std::endl;

    Coll::iterator iter = coll_.find(lineNum);
    assert(iter != coll_.end());
    HistoryPtr &history = iter->second;
    assert(history);

    history->Add(beam);

    // output if all hyps is eod
    bool end = true;
    for (size_t hypoInd = 0; hypoInd <  beam.size(); ++hypoInd) {
      const HypothesisPtr &hypo = beam.at(hypoInd);
      if (hypo->GetWord() != EOS_ID) {
        end = false;
        break;
      }
    }

    if (end) {
      //std::cerr << "beam.size() == 1=" << lineNum << std::endl;
      history->Output(god);
      coll_.erase(iter);
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
    assert(history);

    HypothesisPtr hypo = history->front().at(0);
    hypos.push_back(hypo);
  }

  return hypos;
}

void Histories::OutputRemaining(const God &god) const
{
  for (const Coll::value_type &ele: coll_) {
    const HistoryPtr &history = ele.second;
    assert(history);

    history->Output(god);
  }
}


} // namespace

