#pragma once

#include <map>
#include <numeric>
#include <boost/timer/timer.hpp>

#include "common/scorer.h"
#include "common/exception.h"
#include "common/god.h"
#include "common/utils.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/nth_element.h"

#include "gpu/decoder/encoder_decoder.h"

namespace amunmt {
namespace GPU {

class BestHyps : public BestHypsBase
{
  public:
    BestHyps(const BestHyps &copy) = delete;
    BestHyps(const God &god);

    void DisAllowUNK(mblas::Matrix& Prob) {
      SetColumn(Prob, UNK_ID, std::numeric_limits<float>::lowest());
    }

    void FindBests(const BeamSize& beamSizes,
                   mblas::Matrix& Probs,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst) {
      nthElement_.getNBestList(beamSizes, Probs, outCosts, outKeys, isFirst);
    }

    std::vector<SoftAlignmentPtr> GetAlignments(Scorer& scorer, size_t hypIndex);

    std::vector<SoftAlignmentPtr> GetAlignments(const std::vector<ScorerPtr>& scorers,
                                                size_t hypIndex);
    void CalcBeam(
        const Hypotheses& prevHyps,
        Scorer& scorer,
        const Words& filterIndices,
        Beams &beams,
        const BeamSize &beamSizes);


    void CalcBeam(
        const Hypotheses& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        Beams &beams,
        const BeamSize &beamSizes);

  private:
    NthElement nthElement_;
    DeviceVector<unsigned> keys;
    DeviceVector<float> Costs;
};

}
}

