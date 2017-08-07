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

void BeamSize::Init(uint maxBeamSize, EncOutPtr encOut)
{
  const Sentences &sentences = encOut->GetSentences();

  total_ = sentences.size();

  sizes_.clear();
  sizes_.resize(sentences.size());
  for (size_t i = 0; i < sentences.size(); ++i) {
    Element ele(i * maxBeamSize, 1);
    sizes_[i] = ele;
  }

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
  for (Element& beamSize : sizes_) {
    beamSize.second = val;
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
  assert(sizes_[ind].second > 0);
  --sizes_[ind].second;
  --total_;
}

std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << "sizes_=";
  for (size_t i = 0; i < sizes_.size(); ++i) {
    const Element &ele = sizes_[i];
    strm << "(" << ele.first << "," << ele.second << ") ";
  }

  strm << " sentences_=" << sentences_.size();

  return strm.str();
}

}


