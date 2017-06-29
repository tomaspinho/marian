#pragma once

#include <memory>
#include "sentences.h"

namespace amunmt {

class God;
class Histories;

void TranslationTaskAndOutput(const God &god, SentencesPtr sentences);
std::shared_ptr<Histories> TranslationTask(const God &god, SentencesPtr sentences);
void TranslateMaxiBatchAndOutput(God &god, SentencesPtr maxiBatch, size_t miniSize, int miniWords);

}  // namespace amunmt
