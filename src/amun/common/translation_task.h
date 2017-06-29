#pragma once

#include <memory>
#include "sentences.h"

namespace amunmt {

class God;
class Histories;

class TranslationTask
{
public:
  void RunAndOutput(const God &god, SentencesPtr sentences);
  std::shared_ptr<Histories> Run(const God &god, SentencesPtr sentences);
  void RunMaxiBatchAndOutput(God &god, SentencesPtr maxiBatch, size_t miniSize, int miniWords);

};

}  // namespace amunmt
