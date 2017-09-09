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

    std::vector<SoftAlignmentPtr> GetAlignments(const BaseMatrix &attention, const Scorer& scorer, size_t hypIndex);

    void CalcBeam(
        const Hypotheses& prevHyps,
        BaseMatrix &probs,
        const BaseMatrix &attention,
        const Scorer& scorer,
        const Words& filterIndices,
        Beams &beams,
        const BeamSize &beamSizes);


  private:
    NthElement nthElement_;
    DeviceVector<float> Costs;
};

}
}

