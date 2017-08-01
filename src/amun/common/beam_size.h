#pragma once
#include <vector>
#include "sentences.h"

namespace amunmt {

class BeamSize
{
public:
  BeamSize(SentencesPtr sentences);

  void Init(SentencesPtr sentences);

  void Set(uint val);

  size_t size() const
  { return sizes_.size(); }

  uint GetTotal() const;

  void Decr(size_t ind);

  uint Get(size_t ind) const
  { return sizes_.at(ind); }

  SentencePtr GetSentence(size_t ind) const
  { return sentences_.at(ind); }

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<uint> sizes_;
  std::vector<SentencePtr> sentences_;

  uint total_;

};

}

