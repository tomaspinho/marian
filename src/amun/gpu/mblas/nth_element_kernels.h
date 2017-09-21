#include "matrix_wrapper.h"

namespace amunmt {
namespace GPU {

template<typename T>
struct NthOut
{
  uint ind;
  T score;

  __device__ __host__
  NthOut() {}

  __device__ __host__
  NthOut(uint init)
  :ind(init)
  ,score(init)
  {}

  __device__ __host__
  NthOut(uint &vInd, float vScore)
  :ind(vInd)
  ,score(vScore)
  {}

  __device__
  NthOut& operator+=(const NthOut& rhs)
  {
    ind += rhs.ind;
    score += rhs.score;
    return *this;
  }

  template<typename T2>
  __device__
  NthOut<T>& operator=(const NthOut<T2>& rhs)
  {
    ind = rhs.ind;
    score = rhs.score;
    return *this;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline std::ostream& operator<<(std::ostream &out, const NthOut<T> &obj)
{
  out << "(" << obj.ind << "," << "obj.score" << ")";
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////

__global__ void gMaxElement(mblas::MatrixWrapper<NthOut<float> > out,
                            const mblas::MatrixWrapper<float> probsWrap,
                            const mblas::MatrixWrapper<uint> batchPositionWrap,
                            uint numBatches);

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOut<float> > out,
                                  mblas::MatrixWrapper<float> probsWrap,
                                  mblas::MatrixWrapper<NthOut<float> > resNewWrap,
                                  const mblas::MatrixWrapper<uint> batchPositionWrap,
                                  const mblas::MatrixWrapper<uint> cumBeamSizesWrap,
                                  uint numBlocks);

__global__ void gGetValueByKey(mblas::MatrixWrapper<float> out,
                              const   mblas::MatrixWrapper<float> in,
                              uint* indices, uint n);

/////////////////////////////////////////////////////////////////////////////////////////

__global__ void gMaxElement(mblas::MatrixWrapper<NthOut<half>> out,
                            const mblas::MatrixWrapper<half> probsWrap,
                            const mblas::MatrixWrapper<uint> batchPositionWrap,
                            uint numBatches);

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOut<half>> out,
                                  mblas::MatrixWrapper<half> probsWrap,
                                  mblas::MatrixWrapper<NthOut<half>> resNewWrap,
                                  const mblas::MatrixWrapper<uint> batchPositionWrap,
                                  const mblas::MatrixWrapper<uint> cumBeamSizesWrap,
                                  uint numBlocks);

__global__ void gGetValueByKey(mblas::MatrixWrapper<half> out,
                              const   mblas::MatrixWrapper<half> in,
                              uint* indices, uint n);



}
}
