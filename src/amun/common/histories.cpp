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
      const Sentence &sentence = hypo->GetSentence();

      if (hypo->GetNumWords() < sentence.size() * 3 &&  hypo->GetWord() != EOS_ID) {
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
  //cerr << "batchSize=" << batchSize << endl;

  std::pair<Hypotheses, std::vector<uint> > ret;
  Hypotheses &survivors = ret.first;
  std::vector<uint> &completed = ret.second;

  for (size_t batchId = 0; batchId < batchSize; ++batchId) {
    const Sentence &sentence = beamSizes_->GetSentence(batchId);
    size_t lineNum = sentence.GetLineNum();

    std::pair<bool, BeamPtr> beamPair = beams.Get(lineNum);

    if (beamPair.first) {
      assert(beamPair.second);
      for (const HypothesisPtr& hypo : *beamPair.second) {
        if (hypo->GetNumWords() < sentence.size() * 3 && hypo->GetWord() != EOS_ID) {
          survivors.push_back(hypo);
        } else {
          //beamSizes_->Decr(batchId);
          beamSizes_->Decr2(lineNum);

          if (beamSizes_->Get2(lineNum).size == 0) {
            completed.push_back(batchId);
          }
        }
      }
    }
  }


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

void Histories::SetBeamSize(uint val)
{
  beamSizes_->Set(val);
}



} // namespace

