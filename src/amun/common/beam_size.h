#pragma once
#include <vector>
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
  // 1st = start ind, 2nd = size
  typedef std::pair<uint, uint> Element;

  BeamSize();
  virtual ~BeamSize();

  virtual void Init(uint maxBeamSize, EncOutPtr encOut);

  void Set(uint val);

  size_t size() const
  { return sentences_.size(); }

  uint GetTotal() const;

  uint GetMaxLength() const
  { return maxLength_; }

  void Decr(size_t ind);

  const SentenceElement &Get(size_t ind) const
  { return sentences_.at(ind); }

  const Sentence &GetSentence(size_t ind) const;

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<Element> sizes_;
  std::vector<SentenceElement> sentences_;

  uint total_;
  uint maxLength_;
};

}

