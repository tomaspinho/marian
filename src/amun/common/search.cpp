#include <boost/timer/timer.hpp>
#include "common/search.h"
#include "common/sentences.h"
#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"

using namespace std;

namespace amunmt {

Search::Search(const God &god)
  : deviceInfo_(god.GetNextDevice()),
    scorers_(god.GetScorers(deviceInfo_, *this)),
    filter_(god.GetFilter()),
    maxBeamSize_(god.Get<size_t>("beam-size")),
    normalizeScore_(god.Get<bool>("normalize")),
    bestHyps_(god.GetBestHyps(deviceInfo_))
{}


Search::~Search() {
#ifdef CUDA
  if (deviceInfo_.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo_.deviceId);
  }
#endif

  // scorers must be deleted 1st. async decoding
  scorers_.clear();
}

void Search::CleanAfterTranslation()
{
  for (auto scorer : scorers_) {
    scorer->CleanUpAfterSentence();
  }
}


void Search::Translate(const God &god, const SentencesPtr sentences)
{
  assert(sentences.get());
  assert(scorers_.size() == 1);

  Scorer &scorer = *scorers_[0];

  scorer.Encode(sentences);
  //scorer.Decode(god);
}

void Search::Encode(const SentencesPtr sentences)
{
  for (auto& scorer : scorers_) {
    scorer->Encode(sentences);
  }
}

States Search::BeginSentenceState(const Sentences& sentences)
{
  States states;
  for (auto& scorer : scorers_) {
    auto state = scorer->NewState();
    scorer->BeginSentenceState(*state, sentences.size());
    states.emplace_back(state);
  }
  return states;
}


States Search::NewStates() const {
  States states;
  for (auto& scorer : scorers_) {
    states.emplace_back(scorer->NewState());
  }
  return states;
}

void Search::FilterTargetVocab(const Sentences& sentences) {
  size_t vocabSize = scorers_[0]->GetVocabSize();
  std::set<Word> srcWords;
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence& sentence = *sentences.at(i);
    for (const auto& srcWord : sentence.GetWords()) {
      srcWords.insert(srcWord);
    }
  }

  filterIndices_ = filter_->GetFilteredVocab(srcWords, vocabSize);
  for (auto& scorer : scorers_) {
    scorer->Filter(filterIndices_);
  }
}

///////////////////////////////////////////////////////////////////////////////////


}

