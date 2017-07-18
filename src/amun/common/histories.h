#pragma once

#include <queue>
#include <algorithm>
#include <map>

#include "history.h"
#include "hypothesis.h"

namespace amunmt {

class Sentences;
class God;

class Histories {
  typedef std::unordered_map<size_t, HistoryPtr> Coll;
  //typedef std::map<size_t, HistoryPtr> Coll;

public:
    Histories() {} // for all histories in translation task
    Histories(const Sentences& sentences, bool normalizeScore);

    /*
    //! iterators
    typedef Coll::iterator iterator;
    typedef Coll::const_iterator const_iterator;

    const_iterator begin() const {
      return coll_.begin();
    }
    const_iterator end() const {
      return coll_.end();
    }
    */

    size_t size() const {
      return coll_.size();
    }

    void AddAndOutput(const God &god, const Beams& beams);

    Hypotheses GetFirstHyps() const;

    void OutputRemaining(const God &god) const;

protected:
    Coll coll_;
    Histories(const Histories &) = delete;
};

///////////////////////////////////////////////////////////////////////////////////////

typedef std::shared_ptr<Histories> HistoriesPtr;

} // namespace


