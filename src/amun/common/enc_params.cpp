#include "enc_params.h"

namespace amunmt {

void EncParams::SetSentences(const SentencesPtr sentences)
{
  assert(sentences.get());
  sentences_ = sentences;

}

}
