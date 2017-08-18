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
    const Sentence &sentence = sentences.Get(i);
    size_t lineNum = sentence.GetLineNum();

    //cerr << "BeamSize lineNum=" << lineNum << " " << sentence.GetLineNum() << endl;

    SentenceElement &ele = (sentences_[i] = SentenceElement(encOut, i, 1, i));
    sentencesMap_[lineNum] = &ele;

    if (sentence.size() > maxLength_) {
      maxLength_ = sentence.size();
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

//const BeamSize::SentenceElement &BeamSize::Get(size_t ind) const
//{ return sentences_.at(ind); }

const BeamSize::SentenceElement &BeamSize::GetOnly() const
{
  assert(sentences_.size() == 1);
  return sentences_[0];
}

const Sentence &BeamSize::GetSentence(size_t ind) const
{
  const SentenceElement &ele = Get(ind);
  return ele.GetSentence();
}

const BeamSize::SentenceElement &BeamSize::Get(size_t ind) const
{
  assert(ind < sentences_.size());
  return sentences_[ind];
}


void BeamSize::Decr(size_t ind)
{
  assert(ind < sentences_.size());
  SentenceElement &ele = sentences_[ind];
  ele.Decr();

  --total_;
}

void BeamSize::DeleteEmpty()
{
  size_t i = 0;
  while (i < sentences_.size()) {
    const SentenceElement &ele = sentences_[i];
    if (ele.size) {
      ++i;
    }
    else {
      sentences_.erase(sentences_.begin() + i);
    }
  }
}

std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << "sentences_=" << sentences_.size();

  if (verbosity) {
    for (size_t i = 0; i < sentences_.size(); ++i) {
      const SentenceElement &ele = sentences_[i];
      strm << " (" << ele.sentenceInd << "," << ele.size << ")";
    }
  }

  return strm.str();
}

///////////////////////////////////////////////////////////////////////////////////
const BeamSize::SentenceElement &BeamSize::GetByLineNum(uint lineNum) const
{
  Map::const_iterator iter = sentencesMap_.find(lineNum);
  assert(iter != sentencesMap_.end());
  const SentenceElement *ele = iter->second;
  return *ele;
}

BeamSize::SentenceElement &BeamSize::GetByLineNum(uint lineNum)
{
  Map::iterator iter = sentencesMap_.find(lineNum);
  assert(iter != sentencesMap_.end());
  SentenceElement *ele = iter->second;
  return *ele;
}

void BeamSize::DecrByLineNum(uint lineNum)
{
  SentenceElement &ele = GetByLineNum(lineNum);
  ele.Decr();

  --total_;

}

///////////////////////////////////////////////////////////////////////////////////

}


