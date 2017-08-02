#pragma once
#include "base_matrix.h"

namespace amunmt {

class EncParams
{
public:

  virtual void SetSentences(const SentencesPtr sentences);
  const Sentences &GetSentences() const
  { return *sentences_; }

  virtual BaseMatrix &GetSentenceMask() = 0;
  virtual const BaseMatrix &GetSentenceMask() const = 0;
  virtual BaseMatrix &GetSourceContext() = 0;
  virtual const BaseMatrix &GetSourceContext() const = 0;

  template<class T>
  T &GetSentenceMask2()
  { return static_cast<T&>(GetSentenceMask()); }

  template<class T>
  const T &GetSentenceMask2() const
  { return static_cast<const T&>(GetSentenceMask()); }

  template<class T>
  T &GetSourceContext2()
  { return static_cast<T&>(GetSourceContext()); }

  template<class T>
  const T &GetSourceContext2() const
  { return static_cast<const T&>(GetSourceContext()); }

protected:
  SentencesPtr sentences_;

};

typedef std::shared_ptr<EncParams> EncParamsPtr;

}
