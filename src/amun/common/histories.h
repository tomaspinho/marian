#pragma once

#include <queue>
#include <algorithm>
#include <map>

#include "history.h"
#include "hypothesis.h"
#include "enc_out.h"

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

    void Init(uint maxBeamSize, EncOutPtr encOut);

    void Add(const Sentence &sentence);

    size_t size() const {
      return coll_.size();
    }

    // 1st = survivors, 2nd = completed sentence index
    std::pair<Hypotheses, std::vector<uint> > AddAndOutput(const God &god, const Beams& beams);

    Hypotheses GetFirstHyps() const;

    void SetNewBeamSize(uint val);
    void SetFirst(bool val);

    const BeamSize &GetBeamSizes() const
    { return *beamSizes_; }

    BeamSize &GetBeamSizes()
    { return *beamSizes_; }

protected:
    Coll coll_;
    BeamSize *beamSizes_;
    bool normalizeScore_;

    Histories(const Histories &) = delete;
};

///////////////////////////////////////////////////////////////////////////////////////

typedef std::shared_ptr<Histories> HistoriesPtr;

} // namespace


