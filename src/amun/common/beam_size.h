#pragma once
#include <vector>
#include "sentences.h"

namespace amunmt {

class BeamSize : public std::vector<uint>
{
public:
  BeamSize(SentencesPtr sentences);

  void Init(uint val);

  /*

  size_t size() const
  { return vec_.size(); }


  uint &at(size_t ind)
  { return vec_.at(ind); }

  const uint &at(size_t ind) const
  { return vec_.at(ind); }

  uint GetTotal() const
  { return total_; }

  std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<uint> vec_;
*/
  uint total_;

};

}

