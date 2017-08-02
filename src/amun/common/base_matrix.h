#pragma once

#include <string>
#include <vector>
#include <memory>
#include "common/types.h"
#include "common/sentences.h"

namespace amunmt {

const size_t SHAPE_SIZE = 4;

class BaseMatrix {
  public:
	BaseMatrix() {}
    virtual ~BaseMatrix() {}

    virtual size_t dim(size_t i) const = 0;

    virtual size_t size() const;

    bool empty() const {
      return size() == 0;
    }

    virtual void Resize(size_t rows, size_t cols, size_t beam = 1, size_t batches = 1) = 0;

    virtual std::string Debug(size_t verbosity = 1) const;
};

/////////////////////////////////////////////////////////////////////////

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

