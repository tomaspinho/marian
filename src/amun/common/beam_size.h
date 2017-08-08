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

    const Sentence &GetSentence() const
    {
      const Sentences &sentences = encOut->GetSentences();
      const SentencePtr &sentence = sentences.at(sentenceInd);
      return *sentence;
    }
  };

public:
  typedef std::unordered_map<size_t, SentenceElement*> Coll;

  typedef Coll::const_iterator const_iterator;

  const_iterator begin() const
  { return sentences2_.begin(); }

  const_iterator end() const
  { return sentences2_.end(); }

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

  void Decr2(size_t lineNum);

  ///////////////////////////////////////////////////////////////////////////

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<SentenceElement> sentences_;
  Coll sentences2_;

  uint total_;
  uint maxLength_;

  SentenceElement &Get3(size_t lineNum);
};

}

