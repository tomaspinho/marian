#pragma once

#include <functional>
#include <vector>
#include <map>

#include "common/types.h"
#include "common/beam_size.h"
#include "scorer.h"

namespace amunmt {

class BestHypsBase
{
  public:
    BestHypsBase(
        bool forbidUNK,
        bool returnNBestList,
        bool isInputFiltered,
        bool returnAttentionWeights,
        const std::map<std::string, float>& weights)
    : forbidUNK_(forbidUNK),
      returnNBestList_(returnNBestList),
      isInputFiltered_(isInputFiltered),
      returnAttentionWeights_(returnAttentionWeights),
      weights_(weights)
    {}

    BestHypsBase(const BestHypsBase&) = delete;

    virtual void CalcBeam(
        const Hypotheses& prevHyps,
        BaseMatrix &probs,
        const BaseMatrix &attention,
        const Scorer& scorer,
        const Words& filterIndices,
        Beams &beams,
        const BeamSize &beamSizes) = 0;

  protected:
    const bool forbidUNK_;
    const bool returnNBestList_;
    const bool isInputFiltered_;
    const bool returnAttentionWeights_;
    const std::map<std::string, float> weights_;

};

typedef std::shared_ptr<BestHypsBase> BestHypsBasePtr;

}
