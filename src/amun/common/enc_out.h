#pragma once
#include "base_matrix.h"

namespace amunmt {

class EncOut
{
public:
  EncOut(SentencesPtr sentences);

  const Sentences &GetSentences() const
  { return *sentences_; }

  template<class T>
  T &GetSourceContext()
  { return static_cast<T&>(GetSourceContextInternal()); }

  template<class T>
  const T &GetSourceContext() const
  { return static_cast<const T&>(GetSourceContextInternal()); }

  template<class T>
  const T &GetSentenceLengths() const
  { return static_cast<const T&>(GetSentenceLengthsInternal()); }

protected:
  SentencesPtr sentences_;

  virtual BaseMatrix &GetSourceContextInternal() = 0;
  virtual const BaseMatrix &GetSourceContextInternal() const = 0;

  virtual const BaseMatrix &GetSentenceLengthsInternal() const = 0;
};

typedef std::shared_ptr<EncOut> EncOutPtr;

}
