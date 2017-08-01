#pragma once

#include <queue>
#include <algorithm>
#include <map>

#include "history.h"
#include "hypothesis.h"
#include "base_matrix.h"

namespace amunmt {

class BeamSize;
class Sentences;
class God;

class Histories {
  typedef std::unordered_map<size_t, HistoryPtr> Coll;
  //typedef std::map<size_t, HistoryPtr> Coll;
  // 1st = line num, 2nd = history (beams and top) for this particular sentence

public:
    Histories(BeamSize *beamSizes, bool normalizeScore);
    virtual ~Histories();

    void Init(EncParamsPtr encParams);

    size_t size() const {
      return coll_.size();
    }

    Hypotheses AddAndOutput(const God &god, const Beams& beams);

    Hypotheses GetFirstHyps() const;

    void OutputRemaining(const God &god) const;

    void SetBeamSize(uint val);

    const BeamSize &GetBeamSizes() const
    { return *beamSizes_; }

protected:
    Coll coll_;
    BeamSize *beamSizes_;

    Histories(const Histories &) = delete;
};

///////////////////////////////////////////////////////////////////////////////////////

typedef std::shared_ptr<Histories> HistoriesPtr;

} // namespace


