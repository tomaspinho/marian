#include "enc_params_gpu.h"
#include "gpu/mblas/matrix_functions.h"

namespace amunmt {
namespace GPU {
namespace mblas {

EncParamsGPU::EncParamsGPU(SentencesPtr sentences)
:EncParams(sentences)
{
  size_t tab = 0;
  size_t maxSentenceLength = sentences->GetMaxLength();

  //cerr << "1dMapping=" << mblas::Debug(dMapping, 2) << endl;
  HostVector<char> hMapping(maxSentenceLength * sentences->size(), 0);
  for (size_t i = 0; i < sentences->size(); ++i) {
    for (size_t j = 0; j < sentences->at(i)->GetWords(tab).size(); ++j) {
      hMapping[i * maxSentenceLength + j] = 1;
    }
  }

  sentencesMask_.NewSize(maxSentenceLength, sentences->size(), 1, 1);
  mblas::copy(thrust::raw_pointer_cast(hMapping.data()),
              hMapping.size(),
              sentencesMask_.data(),
              cudaMemcpyHostToDevice);

}

}
}
}

