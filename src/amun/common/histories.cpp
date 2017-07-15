#include <iomanip>
#include "histories.h"
#include "sentences.h"
#include "god.h"

namespace amunmt {

Histories::Histories(const Sentences& sentences, bool normalizeScore)
 : coll_(sentences.size())
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i).get();
    size_t lineNum = sentence.GetLineNum();
    History *history = new History(lineNum, normalizeScore, 3 * sentence.size());
    coll_[i].reset(history);
    //coll_[lineNum].reset(history);
  }
}

void Histories::AddAndOutput(const God &god, const Beams& beams)
{
  assert(size() == beams.size());

  for (size_t i = 0; i < size(); ++i) {
    const Beam &beam = beams[i];

    if (beam.empty()) {
      /*
      if (history) {
        history->Output(god);
        history.reset();
      }
      */
    }
    else {
      HistoryPtr &history = coll_[i];
      /*
      size_t lineNum = beam.at(0)->GetLineNum();
      for (size_t beamInd = 1; beamInd < beam.size(); ++beamInd) {
        assert(lineNum == beam.at(beamInd)->GetLineNum());
      }
      std::cerr << "beam=" << beam.size() << " " << lineNum << std::endl;

      Coll::iterator iter = coll_.find(lineNum);
      assert(iter != coll_.end());
      HistoryPtr &history = iter->second;
      */
      assert(history);
      history->Add(beam);
    }
  }
}

Beam Histories::GetFirstHyps() const
{
  Beam beam;
  for (const Coll::value_type &ele: coll_) {
    const HistoryPtr &history = ele.second;
    beam.push_back(history->front().at(0));
  }
  return beam;
}

void Histories::OutputRemaining(const God &god) const
{
  for (const Coll::value_type &ele: coll_) {
    const HistoryPtr &history = ele.second;

    if (history) {
      history->Output(god);
    }
  }
}


} // namespace

