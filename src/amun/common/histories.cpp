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

void Histories::Init(uint maxBeamSize, EncOutPtr encOut)
{
  beamSizes_->Init(maxBeamSize, encOut);
  //cerr << "beamSizes_=" << beamSizes_->Debug(0) << endl;

  const Sentences &sentences = encOut->GetSentences();
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = sentences.Get(i);
    size_t lineNum = sentence.GetLineNum();
    History *history = new History(sentence, normalizeScore_, 3 * sentence.size());
    coll_[lineNum].reset(history);
  }
}

std::pair<Hypotheses, std::vector<uint> > Histories::AddAndOutput(const God &god, const Beams& beams)
{
  assert(size() <= beams.size());

  // beam sizes
  size_t batchSize = beamSizes_->size();
  //cerr << "batchSize=" << batchSize << endl;

  std::pair<Hypotheses, std::vector<uint> > ret;
  Hypotheses &survivors = ret.first;
  std::vector<uint> &completed = ret.second;

  for (size_t batchId = 0; batchId < batchSize; ++batchId) {
    const BeamSize::SentenceElement &ele = beamSizes_->Get(batchId);
    const Sentence &sentence = ele.GetSentence();
    size_t lineNum = sentence.GetLineNum();

    std::pair<bool, const Beam*> beamPair = beams.Get(lineNum);

    if (beamPair.first) {
      const Beam *beam = beamPair.second;
      assert(beam);

      // add new hypos to history
      Coll::iterator iterHist = coll_.find(lineNum);
      assert(iterHist != coll_.end());
      HistoryPtr &history = iterHist->second;
      assert(history);

      history->Add(*beam);

      // see if any output reaches </s> or is over length limit
      for (const HypothesisPtr& hypo : *beam) {
        if (hypo->GetNumWords() < sentence.size() * 3 && hypo->GetWord() != EOS_ID) {
          survivors.push_back(hypo);
        }
        else {
          //beamSizes_->Decr(batchId);
          beamSizes_->Decr(batchId);
        }
      }

      // output if not more hypos
      if (ele.size == 0) {
        completed.push_back(batchId);

        history->Output(god);
        coll_.erase(iterHist);
      }
    }
    else {
      assert(ele.size == 0);
    }
  }

  /*
  for (Beams::const_iterator iter = beams.begin(); iter != beams.end(); ++iter) {
    size_t lineNum = iter->first;
    const Beam *beam = iter->second.get();
    assert(beam);

    // add new hypos to history
    Coll::iterator iterHist = coll_.find(lineNum);
    assert(iterHist != coll_.end());
    HistoryPtr &history = iterHist->second;
    assert(history);

    history->Add(*beam);

    // see if any output reaches </s> or is over length limit
    for (const HypothesisPtr& hypo : *beam) {
      const Sentence &sentence = hypo->GetSentence();
      if (hypo->GetNumWords() < sentence.size() * 3 && hypo->GetWord() != EOS_ID) {
        survivors.push_back(hypo);
      }
      else {
        //beamSizes_->Decr(batchId);
        beamSizes_->Decr(batchId);
      }
    }

    // output if not more hypos
    if (ele.size == 0) {
      completed.push_back(batchId);

      history->Output(god);
      coll_.erase(iterHist);
    }
  }
  */

  return ret;
}

Hypotheses Histories::GetFirstHyps() const
{
  Hypotheses hypos;

  for (size_t i = 0; i < beamSizes_->size(); ++i) {
    const Sentence &sentence = beamSizes_->GetSentence(i);
    size_t lineNum = sentence.GetLineNum();

    Coll::const_iterator iter = coll_.find(lineNum);
    assert(iter != coll_.end());

    const HistoryPtr &history = iter->second;
    assert(history);

    HypothesisPtr hypo = history->front().at(0);
    hypos.push_back(hypo);
  }

  return hypos;
}

void Histories::SetNewBeamSize(uint val)
{
  beamSizes_->SetNewBeamSize(val);
}



} // namespace

