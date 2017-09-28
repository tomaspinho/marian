#include "gru.h"

namespace amunmt {
namespace GPU {

///////////////////////////////////////////////////////////////////////////////////////////////
__device__
half htanh(const half x)
{
  //half ret = ((half)1.0f - hexp((half)-2.0f * x)) / ((half)1.0f + hexp((half)-2.0f * x));
  //half ret = (hexp((half)2.0f * x) - (half)1.0f) / (hexp((half)2.0f * x) + (half)1.0f);
  //half ret = (hexp(x) - hexp(-x)) / (hexp(x) + hexp(-x));
  half ret = tanhf(x);

  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gElementwiseOps(mblas::MatrixWrapper<FLOAT> outWrap,
                                const mblas::MatrixWrapper<FLOAT> stateWrap,
                                const mblas::MatrixWrapper<FLOAT> ruhWrap,
                                const mblas::MatrixWrapper<FLOAT> tempWrap,
                                const mblas::MatrixWrapper<FLOAT> bWrap,
                                const mblas::MatrixWrapper<FLOAT> bx1Wrap,
                                const mblas::MatrixWrapper<FLOAT> bx2Wrap)
{
  const uint rows = stateWrap.dim(0);
  const uint cols = stateWrap.dim(1);
  assert(blockIdx.x < rows);
  assert(ruhWrap.dim(1) == cols * 3);

  for(int tid = 0; tid < cols; tid += blockDim.x) {
    int i = tid + threadIdx.x;
    if(i < cols) {
      FLOAT ev1 = hexp(-(ruhWrap(blockIdx.x, i, 0, 0)
                         + bWrap[i]
                         + tempWrap(blockIdx.x, i, 0, 0)
                        )
                      );
      FLOAT r = ((FLOAT)1.0f) / ((FLOAT)1.0f + ev1);

      int k = i + cols;
      FLOAT ev2 = hexp(-(ruhWrap(blockIdx.x, k, 0, 0)
                         + bWrap[k]
                         + tempWrap(blockIdx.x, k, 0, 0)
                        )
                      );
      FLOAT u = ((FLOAT)1.0f) / ((FLOAT)1.0f + ev2);

      FLOAT hv = ruhWrap(blockIdx.x, 2*cols + i, 0, 0)
               + bx1Wrap[i];

      FLOAT t2v = tempWrap(blockIdx.x, 2*cols + i, 0, 0)
                + bx2Wrap[i];

      hv = htanh(hv + r * t2v);
      //hv = tanhf(hv + r * t2v);
      outWrap(blockIdx.x, i, 0, 0) = ((FLOAT)1.0f - u) * hv + u * stateWrap(blockIdx.x, i, 0, 0);
    }
  }
}

}
}

