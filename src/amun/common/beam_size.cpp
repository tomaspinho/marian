#include <sstream>
#include "beam_size.h"
#include "utils.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize()
:maxLength_(0)
{
}

BeamSize::~BeamSize()
{}

void BeamSize::Init(EncOutPtr encOut)
{
  const Sentences &sentences = encOut->GetSentences();

  sizes_.clear();
  sizes_.resize(sentences.size(), 1);
  total_ = sentences.size();

  maxLength_ = 0;
  sentences_.resize(sentences.size());

  for (size_t i = 0; i < sentences.size(); ++i) {
    SentenceElement ele = {encOut, i};
    sentences_[i] = ele;

    const Sentences &sentences = encOut->GetSentences();
    SentencePtr sentence = sentences.at(i);
    if (sentence->size() > maxLength_) {
      maxLength_ = sentence->size();
    }
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
  const EncOutPtr &encOut = ele.encOut;
  size_t sentenceInd = ele.sentenceInd;

  const Sentences &sentences = encOut->GetSentences();
  const SentencePtr &sentence = sentences.at(sentenceInd);
  return *sentence;
}


void BeamSize::Decr(size_t ind)
{
  assert(sizes_[ind] > 0);
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


