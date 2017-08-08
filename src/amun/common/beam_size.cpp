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

  maxLength_ = 0;
  sentences_.resize(sentences.size());

  for (size_t i = 0; i < sentences.size(); ++i) {
    SentenceElement ele = {encOut, i, i * maxBeamSize, 1};
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
  for (SentenceElement& ele : sentences_) {
    ele.size = val;
  }
  total_ = size() * val;
}

uint BeamSize::GetTotal() const
{
  return total_;
}

const BeamSize::SentenceElement &BeamSize::Get(size_t ind) const
{ return sentences_.at(ind); }

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
  assert(sentences_[ind].size > 0);
  --sentences_[ind].size;

  --total_;
}

std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << "sentences_=";
  for (size_t i = 0; i < sentences_.size(); ++i) {
    const SentenceElement &ele = sentences_[i];
    strm << "(" << ele.sentenceInd << "," << ele.startInd << "," << ele.size << ") ";
  }

  return strm.str();
}

}


