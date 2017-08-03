#pragma once
#include "base_matrix.h"

namespace amunmt {

class EncParams
{
public:
  EncParams(SentencesPtr sentences);

  const Sentences &GetSentences() const
  { return *sentences_; }

  template<class T>
  const T &GetSentenceMask() const
  { return static_cast<const T&>(GetSentenceMaskInternal()); }

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

  virtual const BaseMatrix &GetSentenceMaskInternal() const = 0;

  virtual BaseMatrix &GetSourceContextInternal() = 0;
  virtual const BaseMatrix &GetSourceContextInternal() const = 0;

  virtual const BaseMatrix &GetSentenceLengthsInternal() const = 0;
};

typedef std::shared_ptr<EncParams> EncParamsPtr;

}
