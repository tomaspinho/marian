#include <iomanip>
#include "histories.h"
#include "sentences.h"
#include "god.h"

using namespace std;

namespace amunmt {

Histories::Histories(BeamSize& beamSizes, bool normalizeScore)
:beamSizes_(beamSizes)
{
  for (size_t i = 0; i < beamSizes.size(); ++i) {
    const Sentence &sentence = *beamSizes.GetSentence(i).get();
    size_t lineNum = sentence.GetLineNum();
    History *history = new History(sentence, normalizeScore, 3 * sentence.size());
    coll_[lineNum].reset(history);
  }
}

void Histories::AddAndOutput(const God &god, const Beams& beams)
{
  assert(size() <= beams.size());

  for (const Beams::Coll::value_type &ele: beams) {
    const Beam &beam = *ele.second;
    assert(!beam.empty());

    size_t lineNum = beam.GetLineNum();

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
      history->Output(god);
      coll_.erase(iter);
    }
  }
}

Hypotheses Histories::GetFirstHyps() const
{
  Hypotheses hypos;

  for (size_t i = 0; i < beamSizes_.size(); ++i) {
    SentencePtr sentence = beamSizes_.GetSentence(i);
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

void Histories::InitBeamSize(uint val)
{
  beamSizes_.Init(val);
}



} // namespace

