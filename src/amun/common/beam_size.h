#pragma once
#include <vector>
#include <unordered_map>
#include "sentences.h"
#include "enc_out.h"

namespace amunmt {

class BeamSize
{
  struct SentenceElement
  {
    EncOutPtr encOut;
    size_t sentenceInd;
    uint startInd;
    uint size;
  };

public:
  BeamSize();
  virtual ~BeamSize();

  virtual void Init(uint maxBeamSize, EncOutPtr encOut);

  void Set(uint val);

  size_t size() const
  { return sentences_.size(); }

  uint GetTotal() const;

  uint GetMaxLength() const
  { return maxLength_; }

  const SentenceElement &Get(size_t ind) const;

  const Sentence &GetSentence(size_t ind) const;

  ///////////////////////////////////////////////////////////////////////////
  const SentenceElement &Get2(size_t lineNum) const;
  SentenceElement &Get2(size_t lineNum);

  void Decr2(size_t lineNum);

  ///////////////////////////////////////////////////////////////////////////

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<SentenceElement> sentences_;
  typedef std::unordered_map<size_t, SentenceElement*> Coll;
  Coll sentences2_;

  uint total_;
  uint maxLength_;
};

}

