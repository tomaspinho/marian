#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize()
{
}

BeamSize::~BeamSize()
{}

void BeamSize::Init(EncParamsPtr encParams)
{
  const Sentences &sentences = encParams->GetSentences();

  sizes_.clear();
  sizes_.resize(sentences.size(), 1);
  total_ = sentences.size();

  sentences_.resize(sentences.size());

  for (size_t i = 0; i < sentences.size(); ++i) {
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

const Sentence &BeamSize::GetSentence(size_t ind) const
{
  const SentenceElement &ele = sentences_.at(ind);
  const EncParamsPtr &encParams = ele.encParams;
  size_t sentenceInd = ele.sentenceInd;

  const Sentences &sentences = encParams->GetSentences();
  const SentencePtr &sentence = sentences.at(sentenceInd);
  return *sentence;
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


