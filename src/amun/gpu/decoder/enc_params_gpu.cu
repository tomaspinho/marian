#include "enc_params_gpu.h"
#include "gpu/mblas/matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {
namespace mblas {

EncParamsGPU::EncParamsGPU(SentencesPtr sentences)
:EncParams(sentences)
{
  size_t tab = 0;
  size_t maxSentenceLength = sentences->GetMaxLength();

  //cerr << "1dMapping=" << mblas::Debug(dMapping, 2) << endl;
  HostVector<uint> hSentenceLengths(sentences->size());
  HostVector<char> hMapping(maxSentenceLength * sentences->size(), 0);

  for (size_t i = 0; i < sentences->size(); ++i) {
    const Sentence &sentence = *sentences->at(i);
    hSentenceLengths[i] = sentence.GetWords(tab).size();

    for (size_t j = 0; j < sentence.GetWords(tab).size(); ++j) {
      hMapping[i * maxSentenceLength + j] = 1;
    }
  }

  sentenceLengths_.NewSize(sentences->size(), 1, 1, 1);
  mblas::copy(thrust::raw_pointer_cast(hSentenceLengths.data()),
              hSentenceLengths.size(),
              sentenceLengths_.data(),
              cudaMemcpyHostToDevice);

  sentencesMask_.NewSize(maxSentenceLength, sentences->size(), 1, 1);
  mblas::copy(thrust::raw_pointer_cast(hMapping.data()),
              hMapping.size(),
              sentencesMask_.data(),
              cudaMemcpyHostToDevice);

  //cerr << "sentenceLengths_=" << sentenceLengths_.Debug(2) << endl;
  //cerr << "sentencesMask_=" << sentencesMask_.Debug(2) << endl;
}

}
}
}

