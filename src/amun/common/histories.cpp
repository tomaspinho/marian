#include <iomanip>
#include "histories.h"
#include "sentences.h"
#include "god.h"

using namespace std;

namespace amunmt {

Histories::Histories(BeamSize *beamSizes, bool normalizeScore)
:beamSizes_(beamSizes)
,normalizeScore_(normalizeScore)
{
}

Histories::~Histories()
{
  delete beamSizes_;
}

void Histories::Init(EncParamsPtr encParams)
{
  beamSizes_->Init(encParams);

  for (size_t i = 0; i < beamSizes_->size(); ++i) {
    const Sentence &sentence = *beamSizes_->GetSentence(i).get();
    size_t lineNum = sentence.GetLineNum();
    History *history = new History(sentence, normalizeScore_, 3 * sentence.size());
    coll_[lineNum].reset(history);
  }
}

Hypotheses Histories::AddAndOutput(const God &god, const Beams& beams)
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

  // beam sizes
  size_t batchSize = beamSizes_->size();
  Hypotheses survivors;
  for (size_t batchId = 0; batchId < batchSize; ++batchId) {
    SentencePtr sentence = beamSizes_->GetSentence(batchId);
    size_t lineNum = sentence->GetLineNum();

    const BeamPtr beam = beams.Get(lineNum);
    //assert(beam);

    if (beam) {
      for (const HypothesisPtr& h : *beam) {
        if (h->GetWord() != EOS_ID) {
          survivors.push_back(h);
        } else {
          beamSizes_->Decr(batchId);
        }
      }
    }
  }

  return survivors;
}

Hypotheses Histories::GetFirstHyps() const
{
  Hypotheses hypos;

  for (size_t i = 0; i < beamSizes_->size(); ++i) {
    SentencePtr sentence = beamSizes_->GetSentence(i);
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

void Histories::SetBeamSize(uint val)
{
  beamSizes_->Set(val);
}



} // namespace

