#include "nth_element_kernels.h"

namespace amunmt {
namespace GPU {


#define SHARED_SIZE 512

/////////////////////////////////////////////////////////////////////////////////

#define UNROLL_MAXARG_LOOP_HALF( n, max ) \
  if (tid < (n) && tid + (n) < ( max ) ) { \
    if (sdataHalf[tid + ( n ) ] > sdataHalf[tid]) { \
      sdataHalf[tid] = sdataHalf[tid + ( n ) ]; \
      indices[tid] = indices[tid + ( n ) ]; \
    } \
  }

__global__ void gMaxElement(mblas::MatrixWrapper<NthOut<half>> out,
                            const mblas::MatrixWrapper<half> probsWrap,
                            const mblas::MatrixWrapper<uint> batchPositionWrap,
                            uint numBatches)
{
  extern __shared__ half sdataHalf[];
  __shared__ uint indices[SHARED_SIZE];

  uint tid = threadIdx.x;

  for (uint batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    uint begin = batchPositionWrap[batchIdx];
    uint end = batchPositionWrap[batchIdx + 1];

    uint i = begin + blockIdx.x * (blockDim.x * 2) + tid;

    sdataHalf[tid] = -65504;

    if (i < end) {
      sdataHalf[tid] = probsWrap[i];
      indices[tid] = i;
    }

    if (i + blockDim.x < end) {
      half a = probsWrap[i];
      half b = probsWrap[i + blockDim.x];
      if (a > b) {
        sdataHalf[tid] = a;
        indices[tid] = i;
      } else {
        sdataHalf[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + 2 * gridDim.x * blockDim.x < end) {
      i += 2 * gridDim.x * blockDim.x;

      half a = probsWrap[i];
      if (a > sdataHalf[tid]) {
        sdataHalf[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < end) {
        half b = probsWrap[i + blockDim.x];
        if (b > sdataHalf[tid]) {
          sdataHalf[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (uint s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < end) {
        if (sdataHalf[tid + s] > sdataHalf[tid]) {
          sdataHalf[tid] = sdataHalf[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP_HALF(32, end);
    UNROLL_MAXARG_LOOP_HALF(16, end);
    UNROLL_MAXARG_LOOP_HALF(8, end);
    UNROLL_MAXARG_LOOP_HALF(4, end);
    UNROLL_MAXARG_LOOP_HALF(2, end);
    UNROLL_MAXARG_LOOP_HALF(1, end);

    if (tid == 0) {
      out[blockIdx.x + batchIdx * gridDim.x] = {indices[0], sdataHalf[0]};
    }
    __syncthreads();
  }
}

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOut<half>> out,
                                  mblas::MatrixWrapper<half> probsWrap,
                                  mblas::MatrixWrapper<NthOut<float>> resNewWrap,
                                  const mblas::MatrixWrapper<uint> batchPositionWrap,
                                  const mblas::MatrixWrapper<uint> cumBeamSizesWrap,
                                  uint numBlocks)
{
  extern __shared__ half sdataHalf[];
  __shared__ uint indices[SHARED_SIZE];
  __shared__ half bestBinCostHalf;
  __shared__ uint bestBinCostIdx;

  const uint tid = threadIdx.x;
  const uint batchIdx = blockIdx.x;
  const uint N = batchPositionWrap[batchIdx + 1] - batchPositionWrap[batchIdx];
  uint num_bins = uint(N / (2 * SHARED_SIZE)) + uint(N % (2 * SHARED_SIZE) != 0);
  //if (num_bins > 500) {
  //  num_bins = 500;
  //}

  for (uint pos = cumBeamSizesWrap[batchIdx]; pos < cumBeamSizesWrap[batchIdx + 1]; ++pos) {
    uint i = tid;

    sdataHalf[tid] = -65504;

    if (i < num_bins) {
      sdataHalf[tid] = out[batchIdx * numBlocks + i].score;
      indices[tid] = i;
    }

    if (i + blockDim.x < num_bins) {
      half a = out[batchIdx * numBlocks + i].score;
      half b = out[batchIdx * numBlocks + i + blockDim.x].score;
      if (a > b) {
        sdataHalf[tid] = a;
        indices[tid] = i;
      } else {
        sdataHalf[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + 2 * blockDim.x < num_bins) {
      i += 2 * blockDim.x;

      half a = out[batchIdx * numBlocks + i].score;
      if (a > sdataHalf[tid]) {
        sdataHalf[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < num_bins) {
        half b = out[batchIdx * numBlocks + i + blockDim.x].score;
        if (b > sdataHalf[tid]) {
          sdataHalf[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (uint s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < num_bins) {
        if (sdataHalf[tid + s] > sdataHalf[tid]) {
          sdataHalf[tid] = sdataHalf[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP_HALF(32, num_bins);
    UNROLL_MAXARG_LOOP_HALF(16, num_bins);
    UNROLL_MAXARG_LOOP_HALF(8, num_bins);
    UNROLL_MAXARG_LOOP_HALF(4, num_bins);
    UNROLL_MAXARG_LOOP_HALF(2, num_bins);
    UNROLL_MAXARG_LOOP_HALF(1, num_bins);

    if (tid == 0) {
      bestBinCostHalf = sdataHalf[0];
      bestBinCostIdx = batchIdx * numBlocks + indices[0];

      probsWrap[ out[bestBinCostIdx].ind ] = -65504;

      resNewWrap[pos].ind = out[bestBinCostIdx].ind;
      resNewWrap[pos].score = bestBinCostHalf;
    }

    __syncthreads();

    i = batchPositionWrap[batchIdx] + (bestBinCostIdx - batchIdx * numBlocks) * (blockDim.x * 2) + tid;
    const uint dist = num_bins * 2 * blockDim.x;

    sdataHalf[tid] = -65504;

    if (i < batchPositionWrap[batchIdx + 1]) {
      sdataHalf[tid] = probsWrap[i];
      indices[tid] = i;
    }

    if (i + blockDim.x < batchPositionWrap[batchIdx + 1]) {
      half a = probsWrap[i];
      half b = probsWrap[i+blockDim.x];
      if (a > b) {
        sdataHalf[tid] = a;
        indices[tid] = i;
      } else {
        sdataHalf[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + dist < batchPositionWrap[batchIdx + 1]) {
      i += dist;

      half a = probsWrap[i];
      if (a > sdataHalf[tid]) {
        sdataHalf[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < batchPositionWrap[batchIdx + 1]) {
        half b = probsWrap[i + blockDim.x];
        if (b > sdataHalf[tid]) {
          sdataHalf[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (uint s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < batchPositionWrap[batchIdx + 1]) {
        if (sdataHalf[tid + s] > sdataHalf[tid]) {
          sdataHalf[tid] = sdataHalf[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP_HALF(32, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP_HALF(16, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP_HALF(8, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP_HALF(4, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP_HALF(2, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP_HALF(1, batchPositionWrap[batchIdx + 1]);

    if (tid == 0) {
      out[bestBinCostIdx] = {indices[0], sdataHalf[0]};
    }
    __syncthreads();
  }
}

__global__ void gGetValueByKey(mblas::MatrixWrapper<half> out,
                              const   mblas::MatrixWrapper<half> in,
                              uint* indices, uint n)
{
  uint tid = threadIdx.x  + blockDim.x * blockIdx.x;
  if (tid < n) {
    uint index = indices[tid];
    out[tid] = in[index];
  }
}

}
}

