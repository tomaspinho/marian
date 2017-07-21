#pragma once
#include <vector>
#include "sentences.h"

namespace amunmt {

class BeamSize
{
public:
  BeamSize(SentencesPtr sentences);

  void Init(uint val);

  size_t size() const
  { return vec_.size(); }

  void Decr(size_t ind);

  uint Get(size_t ind) const
  { return vec_.at(ind); }

  uint GetTotal() const;

  std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<uint> vec_;

  uint total_;

};

}

