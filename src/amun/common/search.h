#pragma once

#include <memory>
#include <set>

#include "common/scorer.h"
#include "common/sentence.h"
#include "common/sentences.h"
#include "common/base_best_hyps.h"
#include "history.h"

namespace amunmt {

class Filter;

class Search {
  public:
    Search(const God &god);
    virtual ~Search();

    void Translate(const God &god, const SentencesPtr sentences);

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

