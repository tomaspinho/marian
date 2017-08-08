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

  void Decr(size_t ind);

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<SentenceElement> sentences_;

  uint total_;
  uint maxLength_;
};

}

