#pragma once

#include <memory>
#include "sentences.h"
#include "history.h"

namespace amunmt {

class God;
class Histories;

class TranslationTask
{
public:
  void RunMaxiBatchAndOutput(God &god, SentencesPtr maxiBatch, size_t miniSize, int miniWords);

protected:
  void RunAndOutput(const God &god, SentencesPtr sentences);
  HistoriesPtr Run(const God &god, SentencesPtr sentences);

};

}  // namespace amunmt
