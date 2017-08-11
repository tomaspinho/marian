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

  template<class T>
  T &GetStates()
  { return static_cast<const T&>(GetStatesInternal()); }

  template<class T>
  const T &GetStates() const
  { return static_cast<const T&>(GetStatesInternal()); }

  template<class T>
  T &GetEmbeddings()
  { return static_cast<const T&>(GetEmbeddingsInternal()); }

  template<class T>
  const T &GetEmbeddings() const
  { return static_cast<const T&>(GetEmbeddingsInternal()); }

protected:
  SentencesPtr sentences_;

  virtual BaseMatrix &GetSourceContextInternal() = 0;
  virtual const BaseMatrix &GetSourceContextInternal() const = 0;

  virtual const BaseMatrix &GetSentenceLengthsInternal() const = 0;

  virtual BaseMatrix &GetStatesInternal() = 0;
  virtual const BaseMatrix &GetStatesInternal() const = 0;

  virtual BaseMatrix &GetEmbeddingsInternal() = 0;
  virtual const BaseMatrix &GetEmbeddingsInternal() const = 0;

};

typedef std::shared_ptr<EncOut> EncOutPtr;

}
