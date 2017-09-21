#include "matrix_wrapper.h"

namespace amunmt {
namespace GPU {

struct NthOut
{
  uint ind;
  float score;

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
};

/////////////////////////////////////////////////////////////////////////////////////////

struct NthOutHalf
{
  uint ind;
  half score;

  __device__ __host__
  NthOutHalf() {}

  __device__ __host__
  NthOutHalf(uint init)
  :ind(init)
  ,score(init)
  {}

  __device__ __host__
  NthOutHalf(uint &vInd, half vScore)
  :ind(vInd)
  ,score(vScore)
  {}

  __device__
  NthOutHalf& operator+=(const NthOutHalf& rhs)
  {
    ind += rhs.ind;
    score += rhs.score;
    return *this;
  }

  __device__
  NthOutHalf& operator=(const NthOut& rhs)
  {
    ind = rhs.ind;
    score = rhs.score;
    return *this;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////

inline std::ostream& operator<<(std::ostream &out, const NthOut &obj)
{
  out << "(" << obj.ind << "," << obj.score << ")";
  return out;
}

inline std::ostream& operator<<(std::ostream &out, const NthOutHalf &obj)
{
  out << "(" << obj.ind << "," << "obj.score" << ")";
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////

__global__ void gMaxElement(mblas::MatrixWrapper<NthOut> out,
                            const mblas::MatrixWrapper<float> probsWrap,
                            const mblas::MatrixWrapper<uint> batchPositionWrap,
                            uint numBatches);

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOut> out,
                                  mblas::MatrixWrapper<float> probsWrap,
                                  mblas::MatrixWrapper<NthOut> resNewWrap,
                                  const mblas::MatrixWrapper<uint> batchPositionWrap,
                                  const mblas::MatrixWrapper<uint> cumBeamSizesWrap,
                                  uint numBlocks);

__global__ void gGetValueByKey(mblas::MatrixWrapper<float> out,
                              const   mblas::MatrixWrapper<float> in,
                              uint* indices, uint n);

/////////////////////////////////////////////////////////////////////////////////////////

__global__ void gMaxElement(mblas::MatrixWrapper<NthOutHalf> out,
                            const mblas::MatrixWrapper<half> probsWrap,
                            const mblas::MatrixWrapper<uint> batchPositionWrap,
                            uint numBatches);

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOutHalf> out,
                                  mblas::MatrixWrapper<half> probsWrap,
                                  mblas::MatrixWrapper<uint> batchPositionWrap,
                                  mblas::MatrixWrapper<NthOutHalf> resNewWrap,
                                  mblas::MatrixWrapper<uint> cumBeamSizesWrap,
                                  uint numBlocks);

__global__ void gGetValueByKey(mblas::MatrixWrapper<half> out,
                              const   mblas::MatrixWrapper<half> in,
                              uint* indices, uint n);



}
}
