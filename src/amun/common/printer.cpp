#include "printer.h"

namespace amunmt {

void Printer(const God &god, const Histories& histories, std::ostream& out) {
  for (size_t i = 0; i < histories.size(); ++i) {
    const History& history = *histories.at(i).get();
    Printer(god, history, out);
  }
}

void Printer(const God &god, const History& history, std::ostream& out) {
  auto bestTranslation = history.Top();
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
    const NBestList &nbl = history.NBest(god.Get<size_t>("beam-size"));
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
      out << history.GetLineNum() << " ||| " << translation << " |||";

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

std::vector<size_t> GetAlignment(const HypothesisPtr& hypothesis) {
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


std::string GetAlignmentString(const std::vector<size_t>& alignment) {
  std::stringstream alignString;
  alignString << " |||";
  for (size_t wordIdx = 0; wordIdx < alignment.size(); ++wordIdx) {
    alignString << " " << wordIdx << "-" << alignment[wordIdx];
  }
  return alignString.str();
}

std::string GetSoftAlignmentString(const HypothesisPtr& hypothesis) {
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

}

