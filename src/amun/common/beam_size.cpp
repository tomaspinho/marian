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

    cerr << "BeamSize lineNum=" << lineNum << " " << sentence.GetLineNum() << endl;

    SentenceElement &ele = (sentences_[i] = SentenceElement(encOut, i, i * maxBeamSize, 1));
    sentences2_[lineNum] = &ele;

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

const BeamSize::SentenceElement &BeamSize::Get(size_t ind) const
{ return sentences_.at(ind); }

const Sentence &BeamSize::GetSentence(size_t ind) const
{
  const SentenceElement &ele = sentences_.at(ind);
  return ele.GetSentence();
}

///////////////////////////////////////////////////////////////////////////
const BeamSize::SentenceElement &BeamSize::Get2(size_t lineNum) const
{
  Coll::const_iterator iter = sentences2_.find(lineNum);
  assert(iter != sentences2_.end());
  return *iter->second;
}

BeamSize::SentenceElement &BeamSize::Get3(size_t lineNum)
{
  Coll::iterator iter = sentences2_.find(lineNum);
  assert(iter != sentences2_.end());
  return *iter->second;
}

void BeamSize::Decr2(size_t lineNum)
{
  SentenceElement &ele = Get3(lineNum);
  assert(ele.size > 0);
  --ele.size;

  --total_;
}

///////////////////////////////////////////////////////////////////////////

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

