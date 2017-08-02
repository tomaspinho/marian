#include "encoder.h"
#include "common/sentences.h"

using namespace std;

namespace amunmt {
namespace GPU {

Encoder::Encoder(const Weights& model)
: embeddings_(model.encEmbeddings_),
  forwardRnn_(model.encForwardGRU_),
  backwardRnn_(model.encBackwardGRU_)
{
}

std::vector<std::vector<size_t>> GetBatchInput(const Sentences& source, size_t tab, size_t maxLen) {
  std::vector<std::vector<size_t>> matrix(maxLen, std::vector<size_t>(source.size(), 0));

  for (size_t j = 0; j < source.size(); ++j) {
    for (size_t i = 0; i < source.at(j)->GetWords(tab).size(); ++i) {
        matrix[i][j] = source.at(j)->GetWords(tab)[i];
    }
  }

  return matrix;
}

void Encoder::Encode(const Sentences& source, size_t tab,
                     EncParamsPtr &encParams)
{
  size_t maxSentenceLength = source.GetMaxLength();

  //cerr << "1dMapping=" << mblas::Debug(dMapping, 2) << endl;
  HostVector<char> hMapping(maxSentenceLength * source.size(), 0);
  for (size_t i = 0; i < source.size(); ++i) {
    for (size_t j = 0; j < source.at(i)->GetWords(tab).size(); ++j) {
      hMapping[i * maxSentenceLength + j] = 1;
    }
  }

  encParams->GetSentenceMask2<mblas::CMatrix>().NewSize(maxSentenceLength, source.size(), 1, 1);
  mblas::copy(thrust::raw_pointer_cast(hMapping.data()),
              hMapping.size(),
              encParams->GetSentenceMask2<mblas::CMatrix>().data(),
              cudaMemcpyHostToDevice);

  //cerr << "GetContext1=" << context.Debug(1) << endl;
  encParams->GetSourceContext2<mblas::Matrix>().NewSize(maxSentenceLength,
                 forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength(),
                 1,
                 source.size());
  //cerr << "GetContext2=" << context.Debug(1) << endl;

  auto input = GetBatchInput(source, tab, maxSentenceLength);

  for (size_t i = 0; i < input.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back();
    }
    embeddings_.Lookup(embeddedWords_[i], input[i]);
    //cerr << "embeddedWords_=" << embeddedWords_.back().Debug(true) << endl;
  }

  //cerr << "GetContext3=" << context.Debug(1) << endl;
  forwardRnn_.Encode(embeddedWords_.cbegin(),
                         embeddedWords_.cbegin() + maxSentenceLength,
                         encParams->GetSourceContext2<mblas::Matrix>(), source.size(), false);
  //cerr << "GetContext4=" << context.Debug(1) << endl;

  backwardRnn_.Encode(embeddedWords_.crend() - maxSentenceLength,
                          embeddedWords_.crend() ,
                          encParams->GetSourceContext2<mblas::Matrix>(), source.size(), true, &encParams->GetSentenceMask2<mblas::CMatrix>());
  //cerr << "GetContext5=" << context.Debug(1) << endl;
}

}
}

