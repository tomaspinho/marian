#pragma once

#define MAX_THREADS 512
#define MAX_BLOCKS 65535

#include <cmath>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <iostream>

#include "gpu/mblas/thrust_functions.h"
#include "gpu/mblas/matrix.h"
#include "gpu/mblas/matrix_wrapper.h"
#include "gpu/mblas/handles.h"

namespace amunmt {
namespace GPU {
namespace mblas {


template <class M>
void Debug(const M& m, size_t pos = 0, size_t l = 8) {
  std::cerr << m.dim(0) << " " << m.dim(1) << std::endl;
  for(size_t i = 0; i < m.dim(0); ++i) {
    std::cerr << i << ": ";
    for(size_t j = pos; j < m.dim(1) && j < pos + l; ++j) {
      std::cerr << m.GetVec()[i * m.dim(1) + j] << " ";
    }
    std::cerr << " ... ";

    for(size_t j = m.dim(1) - l; j < m.dim(1);  ++j) {
      std::cerr << m.GetVec()[i * m.dim(1) + j] << " ";
    }
    std::cerr << std::endl;
    // if(i == 4)
      // break;
  }
}

template<typename T>
std::string Debug(const DeviceVector<T> &vec, size_t verbosity = 1)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    T sum(0);
    for (size_t i = 0; i < vec.size(); ++i) {
      sum += vec[i];
    }
    strm << " sum=" << sum;
  }

  if (verbosity == 2) {
    for (size_t i = 0; i < vec.size(); ++i) {
      strm << " " << vec[i];
    }
  }

  return strm.str();
}

template<typename T>
std::string Debug(const HostVector<T> &vec, size_t verbosity = 1)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    T sum = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
      sum += vec[i];
    }
    strm << " sum=" << sum;
  }

  if (verbosity == 2) {
    for (size_t i = 0; i < vec.size(); ++i) {
      strm << " " << vec[i];
    }
  }

  return strm.str();
}


template<typename T>
void copy(const T *in, size_t count, T *out,  cudaMemcpyKind kind) {
  HANDLE_ERROR( cudaMemcpyAsync(out, in, count * sizeof(T), kind, CudaStreamHandler::GetStream()) );
}

template<class IteratorT1, class IteratorT2>
void copy(IteratorT1 inBegin, IteratorT1 inEnd, IteratorT2 outBegin) {
  thrust::copy(thrust::cuda::par.on(CudaStreamHandler::GetStream()), inBegin, inEnd, outBegin);
}

template<class IteratorT1, class IteratorT2>
void copy_n(IteratorT1 inBegin, size_t size, IteratorT2 outBegin) {
  thrust::copy_n(thrust::cuda::par.on(CudaStreamHandler::GetStream()), inBegin, size, outBegin);
}

void Fill(Matrix& In, float value=0.0f);

void Zero(Matrix& In);

Matrix& Swap(Matrix& Out, Matrix& In);

void Mean(Matrix& Out, const Matrix& In, const IMatrix &sentenceLengths);

void WeightedMean(Matrix& Out,const Matrix& Weights, const Matrix& In, const DeviceVector<uint>& mapping);

Matrix& Transpose(Matrix& Out, const Matrix& In);

Matrix& Transpose(Matrix& Out);

Matrix& Copy(Matrix& Out, const Matrix& In);

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r = 0,
                 const size_t c = 0);
void PasteRows(Matrix& Out, const Matrix& In, const size_t rowNo, size_t colNo=0);

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r = 0,
                const size_t c = 0);

Matrix& Concat(Matrix& Out, const Matrix& In);

void MapMatrix(Matrix& state,
              const IMatrix &sentenceLengths,
              size_t i);

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<uint>& indices);

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<uint>& indices);

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim);

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

Matrix& Softmax(Matrix& Out,
                const DeviceVector<uint>& batchIds,
                const IMatrix &sentenceLengths,
                size_t batchSize);

Matrix& LogSoftmax(Matrix& Out);

template <class Functor>
__global__ void gBroadcast(Functor functor,
                           MatrixWrapper<float> outWrap,
                           const MatrixWrapper<float> in1Wrap,
                           const MatrixWrapper<float> in2Wrap,
                           const MatrixWrapper<uint> batchMappingWrap)
{
  size_t srcSize = outWrap.dim(2);
  size_t inRows = in2Wrap.dim(0);
  size_t cols  = in1Wrap.dim(1);

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < outWrap.size()) {
    /*
    size_t indices[SHAPE_SIZE];
    outWrap.id2Indices(id, indices);

    int row = id / cols; // len * batch for in1
    int srcId = row % srcSize;  // source pos for in1

    int batchMappingIdx = row / srcSize; // batch for in1
    int batchIdx = batchMappingWrap[batchMappingIdx]; // batch id for in1

    outWrap[id] = functor(in1Wrap(srcId, indices[1], 0, batchIdx),
                          in2Wrap(batchMappingIdx, indices[1], 0, 0) );
    */

    int row = id / cols;
    int stateIdx = id % cols;

    int beamIdx = row / srcSize;
    int srcId = row % srcSize;

    int batchIdx = batchMappingWrap[beamIdx];

    outWrap[id] = functor(in1Wrap[(batchIdx * srcSize + srcId) * cols + stateIdx],
                          in2Wrap[beamIdx * cols + stateIdx]);
  }
}

template <class Functor>
Matrix& Broadcast(Functor functor, Matrix& OutOrig, const Matrix& In, const DeviceVector<uint>& batchMapping, size_t srcSize) {
  size_t sumOfBeamSizes = In.dim(0);

  //size_t rows = srcSize * sumOfBeamSizes;
  size_t cols  = OutOrig.dim(1);

  thread_local static Matrix OutNew;
  OutNew.NewSize(sumOfBeamSizes, cols, srcSize);

  MatrixWrapper<float> outWrap(OutNew);
  const MatrixWrapper<float> in1Wrap(OutOrig);
  const MatrixWrapper<float> in2Wrap(In);
  const MatrixWrapper<uint> batchMappingWrap(batchMapping);

  int threads = MAX_THREADS;
  int blocks  = (OutNew.size() / threads) + 1;

  gBroadcast<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (functor, outWrap, in1Wrap, in2Wrap, batchMappingWrap);

  /*
  std::cerr << "nBlocks=" << blocks << std::endl;
  std::cerr << "nThreads=" << threads << std::endl;
  std::cerr << "outWrap=" << outWrap.Debug() << std::endl;
  std::cerr << "in1Wrap=" << in1Wrap.Debug() << std::endl;
  std::cerr << "in2Wrap=" << in2Wrap.Debug() << std::endl;
  std::cerr << "batchMapping=" << Debug(batchMapping, 2) << std::endl;
  std::cerr << "srcSize=" << srcSize << std::endl;
  std::cerr << "sumOfBeamSizes=" << sumOfBeamSizes << std::endl;
  std::cerr << std::endl;

  HANDLE_ERROR(cudaDeviceSynchronize());
  */

  Swap(OutOrig, OutNew);
  return OutOrig;
}

template <class Functor>
__global__ void gBroadcastVecColumn(Functor functor,
                                    MatrixWrapper<float> outWrap,
                                    const MatrixWrapper<float> inWrap) {
  extern __shared__ float sdataOrig[];

  size_t rows  = outWrap.dim(0);
  size_t cols = outWrap.dim(1);

  MatrixWrapper<float> sdata(sdataOrig, rows);

  if (threadIdx.x == 0) {
    for (int i = 0; i < rows; ++i)
      sdata[i] = inWrap[i];
  }
  __syncthreads();

  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    for (int noRow = 0; noRow < rows; ++noRow) {
      float &val = outWrap(noRow, noColumn, 0, 0);
      val = functor(val, sdata[noRow]);
    }
  }
}

template <class Functor>
Matrix& BroadcastVecColumn(Functor functor, Matrix& Out, const DeviceVector<float>& In) {
  size_t rows  = Out.dim(0);
  size_t cols = Out.dim(1);

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + (cols % threads != 0);

  gBroadcastVecColumn<<<blocks, threads, rows * sizeof(float), CudaStreamHandler::GetStream()>>>
    (functor, outWrap, inWrap);

  return Out;
}

template <class Functor>
__global__ void gBroadcastVec(Functor functor,
                              MatrixWrapper<float> outWrap,
                              const MatrixWrapper<float> inWrap)
{
  size_t cols = outWrap.dim(1);

  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    float vecValue = inWrap(0, noColumn, 0, 0);

    for (int dim0 = 0; dim0 < outWrap.dim(0); ++dim0) {
      for (int dim2 = 0; dim2 < outWrap.dim(2); ++dim2) {
        for (int dim3 = 0; dim3 < outWrap.dim(3); ++dim3) {
          float &val = outWrap(dim0, noColumn, dim2, dim3);
          val = functor(val, vecValue);
        }
      }
    }

  }
}

template <class Functor>
Matrix& BroadcastVec(Functor functor, Matrix& Out, const Matrix& In)
{
  //std::cerr << "Out=" << Out.Debug() << std::endl;
  //std::cerr << "In=" << In.Debug() << std::endl;

  size_t cols = Out.dim(1);

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + (cols % threads != 0);
  const cudaStream_t& stream = CudaStreamHandler::GetStream();

  gBroadcastVec<<<blocks, threads, 0, stream>>>
    (functor, outWrap, inWrap);

  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor,
                         MatrixWrapper<float> outWrap)
{
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < outWrap.size()) {
    outWrap[ind] = functor(outWrap[ind]);
  }
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out)
{
  int threads = MAX_THREADS;
  int blocks  = Out.size() / threads + 1;
  const cudaStream_t& stream = CudaStreamHandler::GetStream();

  MatrixWrapper<float> outWrap(Out);

  gElement<<<blocks, threads, 0, stream>>>
    (functor, outWrap);

  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor,
                         MatrixWrapper<float> outWrap,
                         const MatrixWrapper<float> inWrap)
{
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < outWrap.size()) {
    outWrap[ind] = functor(outWrap[ind], inWrap[ind]);
  }
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In)
{
  assert(Out.size() == In.size());

  int threads = MAX_THREADS;
  int blocks  = Out.size() / threads + 1;
  const cudaStream_t& stream = CudaStreamHandler::GetStream();

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);

  gElement<<<blocks, threads, 0, stream>>>
    (functor, outWrap, inWrap);

  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor,
                         MatrixWrapper<float> outWrap,
                         const MatrixWrapper<float> in1Wrap,
                         const MatrixWrapper<float> in2Wrap)
{
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < outWrap.size()) {
    outWrap[ind] = functor(outWrap[ind], in1Wrap[ind], in2Wrap[ind]);
  }
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In1, const Matrix& In2)
{
  //std::cerr << "Out=" << Out.Debug() << std::endl;
  //std::cerr << "In1=" << In1.Debug() << std::endl;
  //std::cerr << "In2=" << In2.Debug() << std::endl;

  assert(Out.size() == In1.size());
  assert(Out.size() == In2.size());

  int threads = MAX_THREADS;
  int blocks  = Out.size() / threads + 1;
  const cudaStream_t& stream = CudaStreamHandler::GetStream();

  //std::cerr << "Element3=" << Out.Debug(0) << std::endl;
  //std::cerr << "Element3=" << In1.Debug(0) << std::endl;
  //std::cerr << "Element3=" << In2.Debug(0) << std::endl;
  //std::cerr << std::endl;
  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> in1Wrap(In1);
  const MatrixWrapper<float> in2Wrap(In2);
  //std::cerr << "outWrap=" << outWrap.Debug() << std::endl;

  gElement<<<blocks, threads, 0, stream>>>
    (functor, outWrap, in1Wrap, in2Wrap);

  //HANDLE_ERROR( cudaPeekAtLastError() );
  //HANDLE_ERROR( cudaDeviceSynchronize() );
  //HANDLE_ERROR( cudaPeekAtLastError() );

  return Out;
}

void SetColumn(Matrix& In, int noColumn, float value);

void Normalization(Matrix& out, const Matrix& in, const Matrix& alpha, const Matrix& beta,
                   float eps);

void Normalization(Matrix& out, const Matrix& in, const Matrix& alpha, float eps);

/////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void gShrinkMatrix0(const MatrixWrapper<uint> newInd,
                              MatrixWrapper<T> in,
                              MatrixWrapper<T> out)
{
  assert(blockIdx.x < out.dim(0));
  assert(blockIdx.y < out.dim(2));
  assert(blockIdx.z < out.dim(3));

  int cols = out.dim(1);
  uint col = threadIdx.x;

  while (col < cols) {
    uint inInd = newInd[blockIdx.x];
    out(blockIdx.x, col, blockIdx.y, blockIdx.z) = in(inInd, col, blockIdx.y, blockIdx.z);

    col += MAX_THREADS;
  }
}

template<typename T>
__global__ void gShrinkMatrix3(const MatrixWrapper<uint> newInd,
                              MatrixWrapper<T> in,
                              MatrixWrapper<T> out)
{
  assert(blockIdx.x < out.dim(0));
  assert(blockIdx.y < out.dim(2));
  assert(blockIdx.z < out.dim(3));

  int cols = out.dim(1);
  uint col = threadIdx.x;

  while (col < cols) {
    uint inInd = newInd[blockIdx.z];
    out(blockIdx.x, col, blockIdx.y, blockIdx.z) = in(blockIdx.x, col, blockIdx.y, inInd);

    col += MAX_THREADS;
  }

}

template<typename T>
uint NewDim(uint dim, const TMatrix<T> &matrix, uint whichDim, uint sizeShrink, uint maxLenDim, uint maxLen)
{
  if (dim == whichDim) {
    return matrix.dim(dim) - sizeShrink;
  }
  else if (dim == maxLenDim) {
    return maxLen;
  }
  else {
    return matrix.dim(dim);
  }
}

template<typename T>
void ShrinkMatrix(TMatrix<T> &matrix,
                  uint whichDim,
                  size_t sizeShrink,
                  const DeviceVector<uint> &newInd,
                  uint maxLenDim = 999,
                  uint maxLen = 999)
{
  assert(whichDim != maxLenDim);

  //thread_local TMatrix<T> out;
  TMatrix<T> out;
  out.NewSize(NewDim(0, matrix, whichDim, sizeShrink, maxLenDim, maxLen),
              NewDim(1, matrix, whichDim, sizeShrink, maxLenDim, maxLen),
              NewDim(2, matrix, whichDim, sizeShrink, maxLenDim, maxLen),
              NewDim(3, matrix, whichDim, sizeShrink, maxLenDim, maxLen));

  /*
  cerr << "sizeShrink=" << sizeShrink
      << " matrix=" << matrix.Debug(0)
      << " out=" << out.Debug(0) << endl;
  */

  const MatrixWrapper<uint> newIndWrap(newInd);
  MatrixWrapper<T> inWrap(matrix);
  MatrixWrapper<T> outWrap(out);

  int nThreads = std::min(MAX_THREADS, (int)out.dim(1));
  dim3 nBlocks(out.dim(0), out.dim(2), out.dim(3));

  const cudaStream_t &stream = CudaStreamHandler::GetStream();

  if (whichDim == 0) {
    gShrinkMatrix0<<<nBlocks, nThreads, 0, stream>>>(newIndWrap, inWrap, outWrap);
  }
  else if (whichDim == 3) {
    gShrinkMatrix3<<<nBlocks, nThreads, 0, stream>>>(newIndWrap, inWrap, outWrap);
  }
  else {
    assert(false);
  }

  out.swap(matrix);
}

/////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void gCopyMatrix(MatrixWrapper<T> out,
                                const MatrixWrapper<T> in)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < in.size()) {
    uint indices[SHAPE_SIZE];
    in.id2Indices(id, indices);

    out(indices[0], indices[1], indices[2], indices[3])
      = in(indices[0], indices[1], indices[2], indices[3]);
  }

}

template<typename T>
void CopyMatrix(TMatrix<T> &out, const TMatrix<T> &in)
{
  if (in.size() == 0) {
    return;
  }
  //cerr << "out=" << out.Debug(0) << endl;
  //cerr << "in=" << in.Debug(0) << endl;

  assert(out.dim(0) >= in.dim(0));
  assert(out.dim(1) >= in.dim(1));
  assert(out.dim(2) >= in.dim(2));
  assert(out.dim(3) >= in.dim(3));

  uint size = in.size();
  uint threads = std::min(size, (uint) MAX_THREADS);
  uint blocks  = (size / threads) + 1;

  const cudaStream_t &stream = CudaStreamHandler::GetStream();
  MatrixWrapper<T> outWrap(out);
  const MatrixWrapper<T> inWrap(in);

  gCopyMatrix<<<blocks, threads, 0, stream>>>(outWrap, inWrap);
}

template<typename T>
uint NewDim2(uint dim, const TMatrix<T> &matrix, uint whichDim, uint sizeEnlarge, uint maxLenDim, uint maxLen)
{
  if (dim == whichDim) {
    return matrix.dim(dim) + sizeEnlarge;
  }
  else if (dim == maxLenDim) {
    return maxLen;
  }
  else {
    return matrix.dim(dim);
  }
}

template<typename T>
void EnlargeMatrix(TMatrix<T> &matrix,
                    uint whichDim, uint val,
                    uint maxLenDim = 999,
                    uint maxLen = 999)
{
  TMatrix<T> out;
  out.NewSize(NewDim2(0, matrix, whichDim, val, maxLenDim, maxLen),
              NewDim2(1, matrix, whichDim, val, maxLenDim, maxLen),
              NewDim2(2, matrix, whichDim, val, maxLenDim, maxLen),
              NewDim2(3, matrix, whichDim, val, maxLenDim, maxLen));

  CopyMatrix(out, matrix);

  out.swap(matrix);
}

/////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void gCopyDimension0(uint whichDim,
                              uint outInd,
                              uint inInd,
                              MatrixWrapper<T> out,
                              const MatrixWrapper<T> in)
{
  assert(whichDim == 0);

  int cols = min(in.dim(1), out.dim(1));
  uint col = threadIdx.x;

  while (col < cols) {
    out(outInd, col, blockIdx.x, blockIdx.y) = in(inInd, col, blockIdx.x, blockIdx.y);

    col += MAX_THREADS;
  }
}

template<typename T>
__global__ void gCopyDimension3(uint whichDim,
                              uint outInd,
                              uint inInd,
                              MatrixWrapper<T> out,
                              const MatrixWrapper<T> in)
{
  assert(whichDim == 3);

  int cols = min(in.dim(1), out.dim(1));
  uint col = threadIdx.x;

  while (col < cols) {
    out(blockIdx.x, col, blockIdx.y, outInd) = in(blockIdx.x, col, blockIdx.y, inInd);

    col += MAX_THREADS;
  }
}

template<typename T>
void CopyDimension(uint whichDim,
                   uint outInd,
                   uint inInd,
                   TMatrix<T> &out,
                   const TMatrix<T> &in)
{
  /*
  std::cerr << "CopyDimension="
            << whichDim << " "
            << outInd << " "
            << inInd << " "
            << out.Debug(0) << " "
            << in.Debug(0) << " "
            << std::endl;
  */
  assert(outInd < out.dim(whichDim));
  assert(inInd < in.dim(whichDim));

  const cudaStream_t &stream = CudaStreamHandler::GetStream();
  MatrixWrapper<T> outWrap(out);
  const MatrixWrapper<T> inWrap(in);

  if (whichDim == 0) {
    int nThreads = std::min(MAX_THREADS, std::min((int)in.dim(1), (int)out.dim(1)));
    dim3 nBlocks(std::min(in.dim(2), out.dim(2)),
                 std::min(in.dim(3), out.dim(3)));

    gCopyDimension0<<<nBlocks, nThreads, 0, stream>>>(whichDim, outInd, inInd, outWrap, inWrap);
  }
  else if (whichDim == 3) {
    int nThreads = std::min(MAX_THREADS, std::min((int)in.dim(1), (int)out.dim(1)));
    dim3 nBlocks(std::min(in.dim(0), out.dim(0)),
                 std::min(in.dim(2), out.dim(2)));

    gCopyDimension3<<<nBlocks, nThreads, 0, stream>>>(whichDim, outInd, inInd, outWrap, inWrap);
  }
  else {
    assert(false);
  }

  //HANDLE_ERROR( cudaStreamSynchronize(stream));
  //HANDLE_ERROR( cudaDeviceSynchronize());
}

/////////////////////////////////////////////////////////////////////////

} // namespace mblas
} // namespace GPU
}
