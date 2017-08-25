#pragma once
#include <cassert>
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
    bool first;

    SentenceElement() {}

    SentenceElement(EncOutPtr vencOut,
                    size_t vsentenceInd,
                    uint vsize)
    {
      //std::cerr << "new SentenceElement=" << vsentenceInd << std::endl;
      encOut = vencOut;
      sentenceInd = vsentenceInd;
      size = vsize;
      first = true;
    }

    void Decr()
    {
      assert(size);
      --size;
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

  void SetNewBeamSize(uint val);
  void SetFirst(bool val);

  size_t size() const
  { return sentences_.size(); }

  uint GetTotal() const;

  uint GetMaxLength() const
  { return maxLength_; }

  //const SentenceElement &Get(size_t ind) const;
  const SentenceElement &GetOnly() const;

  const Sentence &GetSentence(size_t ind) const;

  const SentenceElement &Get(size_t ind) const;
  SentenceElement &Get(size_t ind);

  void Decr(size_t ind);
  void DecrByLineNum(uint lineNum);

  void DeleteEmpty(const std::vector<uint> &completed);

  void AddNewSentences(const std::vector<EncOut::SentenceElement> &newSentences);

  virtual std::string Debug(size_t verbosity = 1) const;

  const SentenceElement &GetByLineNum(uint lineNum) const;
  SentenceElement &GetByLineNum(uint lineNum);

protected:
  std::vector<SentenceElement> sentences_;

  typedef std::unordered_map<size_t, SentenceElement*> Map;
  Map sentencesMap_;

  uint total_;
  uint maxLength_;
};

}

