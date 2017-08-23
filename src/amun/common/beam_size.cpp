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

    SentenceElement &ele = (sentences_[i] = SentenceElement(encOut, i, 1));
    sentencesMap_[lineNum] = &ele;

    if (sentence.size() > maxLength_) {
      maxLength_ = sentence.size();
    }
  }
}

void BeamSize::SetNewBeamSize(uint val)
{
  for (SentenceElement& ele : sentences_) {
    if (ele.first) {
      //std::cerr << "SetNewBeamSize=" << ele.sentenceInd << std::endl;

      total_ += val - ele.size;

      ele.size = val;
      ele.first = false;
    }
  }
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

BeamSize::SentenceElement &BeamSize::Get(size_t ind)
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

void BeamSize::DeleteEmpty(const std::vector<uint> &completed)
{
  std::vector<uint> c2(completed);
  std::sort(c2.rbegin(), c2.rend());
  //cerr << "c2=" << amunmt::Debug(c2, 2) << endl;

  for (size_t i = 0; i < c2.size(); ++i) {
    size_t ind = c2[i];
    const SentenceElement &ele = sentences_[ind];
    assert(ele.size == 0);

    sentences_.erase(sentences_.begin() + ind);
  }
}

void BeamSize::AddNewSentences(const std::vector<EncOut::SentenceElement> &newSentences)
{
  for (size_t i = 0; i < newSentences.size(); ++i) {
    const EncOut::SentenceElement &inEle = newSentences[i];
    SentenceElement outEle(inEle.encOut, inEle.sentenceInd, 1);

    sentences_.push_back(outEle);
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


