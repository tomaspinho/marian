#include <boost/timer/timer.hpp>
#include "common/search.h"
#include "common/sentences.h"
#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"
#include <iomanip>

using namespace std;

namespace amunmt {

Search::Search(const God &god)
  : deviceInfo_(god.GetNextDevice()),
    scorers_(god.GetScorers(deviceInfo_, *this)),
    filter_(god.GetFilter()),
    maxBeamSize_(god.Get<size_t>("beam-size")),
    normalizeScore_(god.Get<bool>("normalize")),
    bestHyps_(god.GetBestHyps(deviceInfo_))
{}


Search::~Search() {
#ifdef CUDA
  if (deviceInfo_.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo_.deviceId);
  }
#endif
}

void Search::CleanAfterTranslation()
{
  for (auto scorer : scorers_) {
    scorer->CleanUpAfterSentence();
  }
}

void Search::TranslateAndOutput(const God &god, const SentencesPtr sentences)
{
  OutputCollector &outputCollector = god.GetOutputCollector();

  std::shared_ptr<Histories> histories = Translate(sentences);

  for (size_t i = 0; i < histories->size(); ++i) {
    const History &history = *histories->at(i);
    size_t lineNum = history.GetLineNum();

    std::stringstream strm;
    Printer(god, history, strm);

    outputCollector.Write(lineNum, strm.str());
  }

}
/*
std::shared_ptr<Histories> Search::Translate(const Sentences& sentences) {
  boost::timer::cpu_timer timer;

  if (filter_) {
    FilterTargetVocab(sentences);
  }

  Encode(sentences);
  States states = BeginSentenceState(sentences);

  States nextStates = NewStates();
  std::vector<uint> beamSizes(sentences.size(), 1);

  std::shared_ptr<Histories> histories(new Histories(sentences, normalizeScore_));
  Beam prevHyps = histories->GetFirstHyps();

  for (size_t decoderStep = 0; decoderStep < 3 * sentences.GetMaxLength(); ++decoderStep) {
    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->Decode(*states[i], *nextStates[i], beamSizes);
    }

    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
        beamSize = maxBeamSize_;
      }
    }
    //cerr << "beamSizes=" << Debug(beamSizes, 1) << endl;

    bool hasSurvivors = CalcBeam(histories, beamSizes, prevHyps, states, nextStates);
    if (!hasSurvivors) {
      break;
    }
  }

  CleanAfterTranslation();

  LOG(progress)->info("Search took {}", timer.format(3, "%ws"));
  return histories;
}
*/

std::shared_ptr<Histories> Search::Translate(const SentencesPtr sentences) {
  boost::timer::cpu_timer timer;
  assert(sentences.get());
  assert(scorers_.size() == 1);

  Scorer &scorer = *scorers_[0];

  scorer.Encode(sentences);

  // begin decoding - create 1st decode states
  State *state = scorer.NewState();
  scorer.BeginSentenceState(*state, sentences->size());

  State *nextState = scorer.NewState();
  std::vector<uint> beamSizes(sentences->size(), 1);

  std::shared_ptr<Histories> histories(new Histories(*sentences, normalizeScore_));
  Beam prevHyps = histories->GetFirstHyps();

  for (size_t decoderStep = 0; decoderStep < 3 * sentences->GetMaxLength(); ++decoderStep) {
    // decode
    scorer.Decode(*state, *nextState, beamSizes);

    // beams
    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
        beamSize = maxBeamSize_;
      }
    }

    size_t batchSize = beamSizes.size();
    Beams beams(batchSize);
    bestHyps_->CalcBeam(prevHyps, scorer, filterIndices_, beams, beamSizes);
    histories->Add(beams);

    Beam survivors;
    for (size_t batchId = 0; batchId < batchSize; ++batchId) {
      for (auto& h : beams[batchId]) {
        if (h->GetWord() != EOS_ID) {
          survivors.push_back(h);
        } else {
          --beamSizes[batchId];
        }
      }
    }

    if (survivors.size() == 0) {
      return histories;
    }

    scorer.AssembleBeamState(*nextState, survivors, *state);

    prevHyps.swap(survivors);

  }

  CleanAfterTranslation();

  LOG(progress)->info("Search took {}", timer.format(3, "%ws"));
  return histories;

}

void Search::Encode(const SentencesPtr sentences)
{
  for (auto& scorer : scorers_) {
    scorer->Encode(sentences);
  }
}

States Search::BeginSentenceState(const Sentences& sentences)
{
  States states;
  for (auto& scorer : scorers_) {
    auto state = scorer->NewState();
    scorer->BeginSentenceState(*state, sentences.size());
    states.emplace_back(state);
  }
  return states;
}

bool Search::CalcBeam(
    std::shared_ptr<Histories>& histories,
    std::vector<uint>& beamSizes,
    Beam& prevHyps,
    States& states,
    States& nextStates)
{
    size_t batchSize = beamSizes.size();
    Beams beams(batchSize);
    bestHyps_->CalcBeam(prevHyps, scorers_, filterIndices_, beams, beamSizes);
    histories->Add(beams);

    Beam survivors;
    for (size_t batchId = 0; batchId < batchSize; ++batchId) {
      for (auto& h : beams[batchId]) {
        if (h->GetWord() != EOS_ID) {
          survivors.push_back(h);
        } else {
          --beamSizes[batchId];
        }
      }
    }

    if (survivors.size() == 0) {
      return false;
    }

    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    prevHyps.swap(survivors);
    return true;
}


States Search::NewStates() const {
  States states;
  for (auto& scorer : scorers_) {
    states.emplace_back(scorer->NewState());
  }
  return states;
}

void Search::FilterTargetVocab(const Sentences& sentences) {
  size_t vocabSize = scorers_[0]->GetVocabSize();
  std::set<Word> srcWords;
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence& sentence = *sentences.at(i);
    for (const auto& srcWord : sentence.GetWords()) {
      srcWords.insert(srcWord);
    }
  }

  filterIndices_ = filter_->GetFilteredVocab(srcWords, vocabSize);
  for (auto& scorer : scorers_) {
    scorer->Filter(filterIndices_);
  }
}

///////////////////////////////////////////////////////////////////////////////////
void Search::Printer(const God &god, const History& history, std::ostream& out) const
{
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

std::vector<size_t> Search::GetAlignment(const HypothesisPtr& hypothesis) const
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


std::string Search::GetAlignmentString(const std::vector<size_t>& alignment)  const
{
  std::stringstream alignString;
  alignString << " |||";
  for (size_t wordIdx = 0; wordIdx < alignment.size(); ++wordIdx) {
    alignString << " " << wordIdx << "-" << alignment[wordIdx];
  }
  return alignString.str();
}

std::string Search::GetSoftAlignmentString(const HypothesisPtr& hypothesis)  const
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


}

