#pragma once
#include <vector>
#include "sentences.h"

namespace amunmt {

class BeamSize : public std::vector<uint>
{
public:
  BeamSize(size_t size, uint val);

  /*
  BeamSize(SentencesPtr sentences);

  size_t size() const
  { return vec_.size(); }

  void Init(uint val);

  uint &at(size_t ind)
  { return vec_.at(ind); }

  const uint &at(size_t ind) const
  { return vec_.at(ind); }

  uint GetTotal() const
  { return total_; }

  std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<uint> vec_;
  uint total_;
*/
};

}

