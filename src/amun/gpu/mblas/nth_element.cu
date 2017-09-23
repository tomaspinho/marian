#include <iostream>
#include "common/utils.h"
#include "matrix_wrapper.h"
#include "nth_element.h"
#include "matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

NthElement::NthElement(uint maxBeamSize, uint maxBatchSize)
: d_breakdown(maxBeamSize, 1, 1, 1)
, maxBeamSize_(maxBeamSize)
, maxBatchSize_(maxBatchSize)
{
  //cerr << "maxBatchSize=" << maxBatchSize << " maxBeamSize=" << maxBeamSize << endl;

  d_batchPosition.reserve(maxBatchSize + 1);
  d_cumBeamSizes.reserve(maxBatchSize + 1);

  d_res.reserve(maxBatchSize * maxBeamSize);
  h_res.reserve(maxBatchSize * maxBeamSize);
}

NthElement::~NthElement()
{
  //cerr << "FOO2" << endl;
}

void NthElement::getNBestList(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
                  std::vector<float>& outCosts, std::vector<uint>& outKeys,
                  const bool isFirst) {
  /*
  cerr << "beamSizes=" << beamSizes.size() << endl;
  cerr << Debug(beamSizes, 2) << endl;
  cerr << "Probs=" << Probs.Debug(0) << endl;
  cerr << "outCosts=" << outCosts.size() << endl;
  cerr << "outKeys=" << outKeys.size() << endl;
  cerr << "isFirst=" << isFirst << endl;
  cerr << endl;
  */
  HostVector<uint> cummulatedBeamSizes(beamSizes.size() + 1);
  HostVector<uint> batchFirstElementIdxs(beamSizes.size() + 1);
  cummulatedBeamSizes[0] = 0;
  batchFirstElementIdxs[0] = 0;

  const uint vocabSize = Probs.dim(1);
  for (uint i = 0; i < beamSizes.size(); ++i) {

    cummulatedBeamSizes[i + 1] = cummulatedBeamSizes[i] + beamSizes[i];
    batchFirstElementIdxs[i + 1] = ((isFirst) ? (i + 1) : cummulatedBeamSizes[i + 1]) * vocabSize;
  }

  uint numHypos = cummulatedBeamSizes.back();
  d_res.NewSize(numHypos, 1, 1, 1);
  h_res.resize(numHypos);

  //cerr << endl;
  //cerr << "numHypos=" << numHypos << endl;
  //cerr << "beamSizes=" << Debug(beamSizes, 2) << endl;
  //cerr << "cummulatedBeamSizes=" << Debug(cummulatedBeamSizes, 2) << endl;
  //cerr << "batchFirstElementIdxs=" << Debug(batchFirstElementIdxs, 2) << endl;
  //cerr << "1Probs=" << Probs.Debug() << endl;

  getNBestList(Probs, batchFirstElementIdxs, cummulatedBeamSizes);

  //cerr << "2Probs=" << Probs.Debug() << endl;
  //cerr << "cummulatedBeamSizes.back()=" << cummulatedBeamSizes.back() << endl;
  //cerr << "cummulatedBeamSizes=" << Debug(cummulatedBeamSizes, 2) << endl;
  GetPairs(numHypos, outKeys, outCosts);

  //cerr << "outCosts=" << Debug(outCosts, 2) << endl;
  //cerr << "outKeys=" << Debug(outKeys, 2) << endl;
}

/////////////////////////////////////////////////////////////////////////////////////

void NthElement::getNBestList(mblas::HalfMatrix &probs,
                              const HostVector<uint>& batchFirstElementIdxs,
                              const HostVector<uint>& cummulatedBeamSizes)
{
  const uint vocabSize = probs.dim(1);
  const uint numBlocks = uint(maxBeamSize_ * vocabSize / (2 * BLOCK_SIZE)) + uint(maxBeamSize_ * vocabSize % (2 * BLOCK_SIZE) != 0);
  const uint numBatches = batchFirstElementIdxs.size() - 1;

  d_out.NewSize(maxBatchSize_ * numBlocks, 1, 1, 1);

  //cerr << "cummulatedBeamSizes=" << cummulatedBeamSizes.size() << endl;
  d_batchPosition.NewSize(batchFirstElementIdxs.size(), 1, 1, 1);
  d_cumBeamSizes.NewSize(cummulatedBeamSizes.size(), 1, 1, 1);
  assert(d_batchPosition.size() == d_cumBeamSizes.size());

  mblas::copy(thrust::raw_pointer_cast(batchFirstElementIdxs.data()),
              batchFirstElementIdxs.size(),
              d_batchPosition.data(),
              cudaMemcpyHostToDevice);
  mblas::copy(thrust::raw_pointer_cast(cummulatedBeamSizes.data()),
              cummulatedBeamSizes.size(),
              d_cumBeamSizes.data(),
              cudaMemcpyHostToDevice);

  mblas::MatrixWrapper<half> probsWrapHalf(probs);
  mblas::MatrixWrapper<uint> batchPositionWrap(d_batchPosition);
  mblas::MatrixWrapper<uint> cumBeamSizesWrap(d_cumBeamSizes);

  mblas::TMatrix<NthOut<half>> outHalf(d_out.dim(0), d_out.dim(1), d_out.dim(2), d_out.dim(3));
  CopyMatrix(outHalf, d_out);
  mblas::MatrixWrapper<NthOut<half>> outWrapHalf(outHalf);

  mblas::TMatrix<NthOut<half>> resHalf(d_res.dim(0), d_res.dim(1), d_res.dim(2), d_res.dim(3));
  CopyMatrix(resHalf, d_res);
  mblas::MatrixWrapper<NthOut<half>> resWrapHalf(resHalf, false);

  gMaxElement<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mblas::CudaStreamHandler::GetStream()>>>
    (outWrapHalf, probsWrapHalf, batchPositionWrap, numBatches);


  gMaxElementUpdate<<<numBatches, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mblas::CudaStreamHandler::GetStream()>>>
      (outWrapHalf,
       probsWrapHalf,
       resWrapHalf,
       batchPositionWrap,
       cumBeamSizesWrap,
       numBlocks);

   CopyMatrix(d_out, outHalf);
   CopyMatrix(d_res, resHalf);

  /*
  CopyMatrix(d_out, outHalf);
  mblas::MatrixWrapper<NthOut<float>> outWrap(d_out);

  mblas::TMatrix<float> probsFloat(probs.dim(0), probs.dim(1), probs.dim(2), probs.dim(3));
  CopyMatrix(probsFloat, probs);
  mblas::MatrixWrapper<float> probsWrap(probsFloat);

  CopyMatrix(d_res, resHalf);
  mblas::MatrixWrapper<NthOut<float>> resWrap(d_res);

  gMaxElementUpdate<<<numBatches, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mblas::CudaStreamHandler::GetStream()>>>
    (outWrap,
     probsWrap,
     resWrap,
     batchPositionWrap,
     cumBeamSizesWrap,
     numBlocks);
  */

  /*
  cerr << "numBlocks=" << numBlocks << endl;
  cerr << "numBatches=" << numBatches << endl;
  cerr << "threads=" << BLOCK_SIZE << endl;

  cerr << "outWrap=" << outWrap.Debug() << endl;

  cerr << "probsWrap=" << probsWrap.Debug() << endl;

  cerr << "batchPositionWrap=" << batchPositionWrap.Debug() << endl;
  cerr << mblas::Debug(d_batchPosition, 2) << endl;

  cerr << "resWrap=" << resWrap.Debug() << endl;
  cerr << mblas::Debug(d_res, 2) << endl;

  cerr << "cumBeamSizesWrap=" << cumBeamSizesWrap.Debug() << endl;
  //cerr << mblas::Debug(d_cumBeamSizes, 2) << endl;

  cerr << endl;
  */

}

void NthElement::getNBestList(mblas::Matrix &probs,
                              const HostVector<uint>& batchFirstElementIdxs,
                              const HostVector<uint>& cummulatedBeamSizes)
{
  /*
  const uint vocabSize = probs.dim(1);
  const uint numBlocks = uint(maxBeamSize_ * vocabSize / (2 * BLOCK_SIZE)) + uint(maxBeamSize_ * vocabSize % (2 * BLOCK_SIZE) != 0);
  const uint numBatches = batchFirstElementIdxs.size() - 1;

  d_out.NewSize(maxBatchSize_ * numBlocks, 1, 1, 1);

  //cerr << "cummulatedBeamSizes=" << cummulatedBeamSizes.size() << endl;
  d_batchPosition.NewSize(batchFirstElementIdxs.size(), 1, 1, 1);
  d_cumBeamSizes.NewSize(cummulatedBeamSizes.size(), 1, 1, 1);
  assert(d_batchPosition.size() == d_cumBeamSizes.size());

  mblas::copy(thrust::raw_pointer_cast(batchFirstElementIdxs.data()),
              batchFirstElementIdxs.size(),
              d_batchPosition.data(),
              cudaMemcpyHostToDevice);
  mblas::copy(thrust::raw_pointer_cast(cummulatedBeamSizes.data()),
              cummulatedBeamSizes.size(),
              d_cumBeamSizes.data(),
              cudaMemcpyHostToDevice);

  mblas::MatrixWrapper<NthOut<float> > outWrap(d_out);
  mblas::MatrixWrapper<float> probsWrap(probs);
  mblas::MatrixWrapper<uint> batchPositionWrap(d_batchPosition);
  mblas::MatrixWrapper<NthOut<float> > resWrap(d_res, false);
  mblas::MatrixWrapper<uint> cumBeamSizesWrap(d_cumBeamSizes);

  gMaxElement<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mblas::CudaStreamHandler::GetStream()>>>
    (outWrap, probsWrap, batchPositionWrap, numBatches);

  gMaxElementUpdate<<<numBatches, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mblas::CudaStreamHandler::GetStream()>>>
    (outWrap,
     probsWrap,
     resWrap,
     batchPositionWrap,
     cumBeamSizesWrap,
     numBlocks);
  */

  mblas::HalfMatrix probsHalf(probs.dim(0), probs.dim(1), probs.dim(2), probs.dim(3));
  CopyMatrix(probsHalf, probs);

  getNBestList(probsHalf, batchFirstElementIdxs, cummulatedBeamSizes);

  CopyMatrix(probs, probsHalf);
}

/////////////////////////////////////////////////////////////////////////////////////

void NthElement::GetPairs(uint number,
                    std::vector<uint>& outKeys,
                    std::vector<float>& outValues)
{
  mblas::copy(d_res.data(), d_res.size(), thrust::raw_pointer_cast(h_res.data()), cudaMemcpyDeviceToHost);
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()) );

  for (uint i = 0; i < number; ++i) {
    outKeys.push_back(h_res[i].ind);
    outValues.push_back(h_res[i].score);
  }
}

void NthElement::getValueByKey(std::vector<float>& out, const mblas::Matrix &d_in) const
{
  // need a model with multiple scorers to test this method
  assert(false);

  mblas::MatrixWrapper<float> breakdownWrap(d_breakdown);
  const mblas::MatrixWrapper<float> inWrap(d_in);

  //gGetValueByKey<<<1, lastN_, 0, stream_>>>
  //  (breakdownWrap, inWrap, h_res_idx, lastN_);

  HANDLE_ERROR( cudaMemcpyAsync(out.data(), d_breakdown.data(), h_res.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, mblas::CudaStreamHandler::GetStream()) );
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
}

}  // namespace GPU
} // namespace amunmt
