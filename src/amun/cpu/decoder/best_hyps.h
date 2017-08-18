#pragma once

#include <vector>
#include <boost/iterator/permutation_iterator.hpp>

#include "common/scorer.h"
#include "common/god.h"
#include "common/exception.h"
#include "cpu/mblas/matrix.h"
#include "cpu/decoder/encoder_decoder.h"

namespace amunmt {
namespace CPU {

class BestHyps : public BestHypsBase
{
  public:
    BestHyps(const God &god);

    void CalcBeam(
        const Hypotheses& prevHyps,
        BaseMatrix &probs,
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
};

}  // namespace CPU
}  // namespace amunmt
