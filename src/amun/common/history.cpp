#include <iomanip>
#include "history.h"
#include "sentences.h"
#include "god.h"

using namespace std;

namespace amunmt {

History::History(const Sentence &sentence, bool normalizeScore, size_t maxLength)
: normalize_(normalizeScore),
sentence_(sentence),
maxLength_(maxLength)
{
  Beam beam(sentence.GetLineNum(), {HypothesisPtr(new Hypothesis(sentence))});
  Add(beam);
}

void History::Add(const Beam& beam)
{
  //std::cerr << "beam=" << beam.size() << " " << GetLineNum()  << " "; //<< std::endl;
  if (beam.back()->GetPrevHyp() != nullptr) {
    for (size_t j = 0; j < beam.size(); ++j) {
      HypothesisPtr hyp = beam.at(j);
      size_t lineNum = hyp->GetLineNum();
      assert(GetLineNum() == lineNum);
      //std::cerr << lineNum << " ";

      if(hyp->GetWord() == EOS_ID || size() == maxLength_ ) {
        float cost = normalize_ ? hyp->GetCost() / history_.size() : hyp->GetCost();
        topHyps_.push({ history_.size(), j, cost });
      }
    }
  }
  history_.push_back(beam);

  //std::cerr << std::endl;
}

NBestList History::NBest(size_t n) const {
  cerr << "NBest1=" << endl;
  NBestList nbest;
  auto topHypsCopy = topHyps_;
  while (nbest.size() < n && !topHypsCopy.empty()) {
    cerr << "NBest2=" << endl;
    auto bestHypCoord = topHypsCopy.top();
    topHypsCopy.pop();

    size_t start = bestHypCoord.i;
    size_t j  = bestHypCoord.j;

    cerr << "NBest3=" << endl;
    Words targetWords;
    HypothesisPtr bestHyp = history_[start].at(j);
    cerr << "NBest4=" << endl;
    while(bestHyp->GetPrevHyp() != nullptr) {
      cerr << "NBest5=" << endl;
      targetWords.push_back(bestHyp->GetWord());
      bestHyp = bestHyp->GetPrevHyp();
    }

    cerr << "NBest6=" << endl;
    std::reverse(targetWords.begin(), targetWords.end());
    nbest.emplace_back(targetWords, history_[bestHypCoord.i].at(bestHypCoord.j));
    cerr << "NBest7=" << endl;
  }
  cerr << "NBest8=" << endl;
  return nbest;
}

Result History::Top() const
{
  cerr << "Top1=" << sentence_.GetLineNum() << " " << history_.size() << endl;
  NBestList nBest(NBest(1));
  assert(nBest.size() == 1);

  cerr << "Top2=" << nBest.size() << endl;
  Result ret(nBest[0]);
  cerr << "Top3" << endl;

  return ret;
}

void History::Output(const God &god) const
{
  std::cerr << "lineNum_=" << GetLineNum() << " ";
  for (size_t i = 0; i < history_.size(); ++i) {
    std::cerr << "(";
    const Beam &beam = history_[i];

    for (size_t j = 0; j < beam.size(); ++j) {
      HypothesisPtr hypo = beam.at(j);
      assert(hypo);

      std::cerr << hypo->GetLineNum() << ",";
    }
    std::cerr << ") ";
  }

  std::stringstream strm;
  std::cerr << "Output1" << std::endl;
  Output(god, strm);
  std::cerr << "Output2" << std::endl;
  std::string str = strm.str();

  std::cerr << "Output3" << std::endl;
  OutputCollector &outputCollector = god.GetOutputCollector();
  std::cerr << "Output4" << std::endl;
  outputCollector.Write(GetLineNum(), str);
  std::cerr << "Output5" << std::endl;

  std::cerr << str << std::endl;
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
  std::cerr << "BAA1" << std::endl;
  auto bestTranslation = Top();
  std::cerr << "BAA2" << std::endl;
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

} // namespace

