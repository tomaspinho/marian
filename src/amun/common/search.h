#pragma once

#include <memory>
#include <set>

#include "common/scorer.h"
#include "common/sentence.h"
#include "common/sentences.h"
#include "common/base_best_hyps.h"

namespace amunmt {

class History;
class Histories;
class Filter;

class Search {
  public:
    Search(const God &god);
    virtual ~Search();

    void TranslateAndOutput(const God &god, const SentencesPtr sentences);
    std::shared_ptr<Histories> Translate(const SentencesPtr sentences);

    void Printer(const God &god, const History& history, std::ostream& out) const;

    size_t MaxBeamSize() const
    { return maxBeamSize_; }

    bool NormalizeScore() const
    { return normalizeScore_; }

    const Words &FilterIndices() const
    { return filterIndices_; }

    BestHypsBasePtr BestHyps() const
    { return bestHyps_; }

  protected:
    States NewStates() const;
    void FilterTargetVocab(const Sentences& sentences);
    void Encode(const SentencesPtr sentences);
    States BeginSentenceState(const Sentences& sentences);

    void CleanAfterTranslation();

    bool CalcBeam(
    		std::shared_ptr<Histories>& histories,
    		std::vector<uint>& beamSizes,
        Beam& prevHyps,
    		States& states,
    		States& nextStates);

    std::vector<size_t> GetAlignment(const HypothesisPtr& hypothesis) const;

    std::string GetAlignmentString(const std::vector<size_t>& alignment) const;
    std::string GetSoftAlignmentString(const HypothesisPtr& hypothesis) const;

    Search(const Search&) = delete;

  protected:
    DeviceInfo deviceInfo_;
    std::vector<ScorerPtr> scorers_;
    std::shared_ptr<const Filter> filter_;
    const size_t maxBeamSize_;
    bool normalizeScore_;
    Words filterIndices_;
    BestHypsBasePtr bestHyps_;
};

}

