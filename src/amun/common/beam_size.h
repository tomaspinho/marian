#pragma once
#include <vector>
#include <unordered_map>
#include <map>
#include "sentences.h"
#include "enc_out.h"

namespace amunmt {

class BeamSize
{

public:
  ///////////////////////////////////////////////////////////////////////////
  struct SentenceElement
  {
    EncOutPtr encOut;
    size_t sentenceInd; // index of the sentence we're translation within encOut.sentences
    uint size;  // beam size 0..beam

    SentenceElement() {}

    SentenceElement(EncOutPtr vencOut,
                    size_t vsentenceInd,
                    uint vsize)
    {
      encOut = vencOut;
      sentenceInd = vsentenceInd;
      size = vsize;
    }

    const Sentence &GetSentence() const
    {
      const Sentences &sentences = encOut->GetSentences();
      const Sentence &sentence = sentences.Get(sentenceInd);
      return sentence;
    }
  };
  ///////////////////////////////////////////////////////////////////////////

  BeamSize();
  virtual ~BeamSize();

  virtual void Init(uint maxBeamSize, EncOutPtr encOut);

  void Set(uint val);

  size_t size() const
  { return sentences_.size(); }

  uint GetTotal() const;

  uint GetMaxLength() const
  { return maxLength_; }

  //const SentenceElement &Get(size_t ind) const;
  const SentenceElement &GetOnly() const;

  const Sentence &GetSentence(size_t ind) const;

  const SentenceElement &Get(size_t ind) const;

  void Decr(size_t ind);

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<SentenceElement> sentences_;

  uint total_;
  uint maxLength_;
};

}

