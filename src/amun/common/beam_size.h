#pragma once
#include <vector>
#include "sentences.h"

namespace amunmt {

class BeamSize
{
public:
  BeamSize(SentencesPtr sentences);

  size_t size() const
  { return vec_.size(); }

  void Init(uint val);

  std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<uint> vec_;

};

}

