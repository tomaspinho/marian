#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize()
{
}

void BeamSize::Init(EncParamsPtr encParams)
{
  sizes_.clear();
  sizes_.resize(encParams->sentences->size(), 1);
  total_ = encParams->sentences->size();

  sentences_.resize(encParams->sentences->size());

  for (size_t i = 0; i < encParams->sentences->size(); ++i) {
    SentenceElement ele = {encParams, i};
    sentences_[i] = ele;
  }
}

void BeamSize::Set(uint val)
{
  for (uint& beamSize : sizes_) {
    beamSize = val;
  }
  total_ = size() * val;
}

uint BeamSize::GetTotal() const
{
  return total_;
}

SentencePtr BeamSize::GetSentence(size_t ind) const
{
  const SentenceElement &ele = sentences_.at(ind);
  const EncParamsPtr &encParams = ele.encParams;
  size_t sentenceInd = ele.sentenceInd;

  SentencesPtr &sentences = encParams->sentences;
  SentencePtr sentence = sentences->at(sentenceInd);
  return sentence;
}


void BeamSize::Decr(size_t ind)
{
  --sizes_[ind];
  --total_;
}

std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << "sizes_=" << amunmt::Debug(sizes_, verbosity);
  strm << " sentences_=" << sentences_.size();

  return strm.str();
}

}


