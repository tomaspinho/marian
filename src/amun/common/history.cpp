#include <iomanip>
#include "history.h"
#include "sentences.h"
#include "god.h"

namespace amunmt {

History::History(size_t lineNo, bool normalizeScore, size_t maxLength)
  : normalize_(normalizeScore),
    lineNum_(lineNo),
   maxLength_(maxLength)
{
  Add({HypothesisPtr(new Hypothesis(lineNo))});
}

void History::Add(const Beam& beam)
{
  std::cerr << "beam=" << beam.size() << " " << lineNum_  << " "; //<< std::endl;
  if (beam.back()->GetPrevHyp() != nullptr) {
    for (size_t j = 0; j < beam.size(); ++j) {
      HypothesisPtr hyp = beam.at(j);
      size_t lineNum = hyp->GetLineNum();
      //assert(lineNum_ == lineNum);
      std::cerr << lineNum << " ";

      if(hyp->GetWord() == EOS_ID || size() == maxLength_ ) {
        float cost = normalize_ ? hyp->GetCost() / history_.size() : hyp->GetCost();
        topHyps_.push({ history_.size(), j, cost });
      }
    }
  }
  history_.push_back(beam);

  std::cerr << std::endl;
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
    HypothesisPtr bestHyp = history_[start].at(j);
    while(bestHyp->GetPrevHyp() != nullptr) {
      targetWords.push_back(bestHyp->GetWord());
      bestHyp = bestHyp->GetPrevHyp();
    }

    std::reverse(targetWords.begin(), targetWords.end());
    nbest.emplace_back(targetWords, history_[bestHypCoord.i].at(bestHypCoord.j));
  }
  return nbest;
}

void History::Output(const God &god) const
{
  //std::cerr << "lineNum_=" << lineNum_ << std::endl;
  std::stringstream strm;

  Output(god, strm);

  OutputCollector &outputCollector = god.GetOutputCollector();
  outputCollector.Write(lineNum_, strm.str());

}

////////////////////////////////////////////////////////////////////////////////////////
// helper functions

std::vector<size_t> GetAlignment(const HypothesisPtr& hypothesis)
{
  std::vector<SoftAlignment> aligns;
  HypothesisPtr last = hypothesis->GetPrevHyp();
  while (last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(*(last->GetAlignment(0)));
    last = last->GetPrevHyp();
  }

  std::vector<size_t> alignment;
  for (auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
    size_t maxArg = 0;
    for (size_t i = 0; i < it->size(); ++i) {
      if ((*it)[maxArg] < (*it)[i]) {
        maxArg = i;
      }
    }
    alignment.push_back(maxArg);
  }

  return alignment;
}


std::string GetAlignmentString(const std::vector<size_t>& alignment)
{
  std::stringstream alignString;
  alignString << " |||";
  for (size_t wordIdx = 0; wordIdx < alignment.size(); ++wordIdx) {
    alignString << " " << wordIdx << "-" << alignment[wordIdx];
  }
  return alignString.str();
}

std::string GetSoftAlignmentString(const HypothesisPtr& hypothesis)
{
  std::vector<SoftAlignment> aligns;
  HypothesisPtr last = hypothesis->GetPrevHyp();
  while (last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(*(last->GetAlignment(0)));
    last = last->GetPrevHyp();
  }

  std::stringstream alignString;
  alignString << " |||";
  for (auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
    alignString << " ";
    for (size_t i = 0; i < it->size(); ++i) {
      if (i>0) alignString << ",";
      alignString << (*it)[i];
    }
    // alternate code: distribute probability mass from alignment to <eos>
    // float aligned_to_eos = (*it)[it->size()-1];
    // for (size_t i = 0; i < it->size()-1; ++i) {
    //  if (i>0) alignString << ",";
    //  alignString << ( (*it)[i] / (1-aligned_to_eos) );
    // }
  }

  return alignString.str();
}
////////////////////////////////////////////////////////////////////////////////////////

void History::Output(const God &god, std::ostream& out) const
{
  auto bestTranslation = Top();
  std::vector<std::string> bestSentenceWords = god.Postprocess(god.GetTargetVocab()(bestTranslation.first));

  std::string best = Join(bestSentenceWords);
  if (god.Get<bool>("return-alignment")) {
    best += GetAlignmentString(GetAlignment(bestTranslation.second));
  }
  if (god.Get<bool>("return-soft-alignment")) {
    best += GetSoftAlignmentString(bestTranslation.second);
  }

  if (god.Get<bool>("n-best")) {
    std::vector<std::string> scorerNames = god.GetScorerNames();
    const NBestList &nbl = NBest(god.Get<size_t>("beam-size"));
    if (god.Get<bool>("wipo")) {
      out << "OUT: " << nbl.size() << std::endl;
    }
    for (size_t i = 0; i < nbl.size(); ++i) {
      const Result& result = nbl[i];
      const Words &words = result.first;
      const HypothesisPtr &hypo = result.second;

      if(god.Get<bool>("wipo")) {
        out << "OUT: ";
      }
      std::string translation = Join(god.Postprocess(god.GetTargetVocab()(words)));
      if (god.Get<bool>("return-alignment")) {
        translation += GetAlignmentString(GetAlignment(bestTranslation.second));
      }
      out << GetLineNum() << " ||| " << translation << " |||";

      for(size_t j = 0; j < hypo->GetCostBreakdown().size(); ++j) {
        out << " " << scorerNames[j] << "= " << std::setprecision(3) << std::fixed << hypo->GetCostBreakdown()[j];
      }

      if(god.Get<bool>("normalize")) {
        out << " ||| " << std::setprecision(3) << std::fixed << hypo->GetCost() / words.size();
      }
      else {
        out << " ||| " << std::setprecision(3) << std::fixed << hypo->GetCost();
      }

      if(i < nbl.size() - 1)
        out << std::endl;
      else
        out << std::flush;

    }
  } else {
    out << best << std::flush;
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////

Histories::Histories(const Sentences& sentences, bool normalizeScore)
 : coll_(sentences.size())
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i).get();
    size_t lineNum = sentence.GetLineNum();
    History *history = new History(lineNum, normalizeScore, 3 * sentence.size());
    coll_[i].reset(history);
    //coll_[lineNum].reset(history);
  }
}

void Histories::AddAndOutput(const God &god, const Beams& beams)
{
  assert(size() == beams.size());

  for (size_t i = 0; i < size(); ++i) {
    const Beam &beam = beams[i];

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
      std::cerr << "beam=" << beam.size() << " " << lineNum << std::endl;

      Coll::iterator iter = coll_.find(lineNum);
      assert(iter != coll_.end());
      HistoryPtr &history = iter->second;
      */
      assert(history);
      history->Add(beam);
    }
  }
}

Beam Histories::GetFirstHyps() const
{
  Beam beam;
  for (const Coll::value_type &ele: coll_) {
    const HistoryPtr &history = ele.second;
    beam.push_back(history->front().at(0));
  }
  return beam;
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

