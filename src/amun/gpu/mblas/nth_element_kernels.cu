#include "nth_element_kernels.h"

namespace amunmt {
namespace GPU {


#define SHARED_SIZE 512

#define UNROLL_MAXARG_LOOP( n, max ) \
  if (tid < (n) && tid + (n) < ( max ) ) { \
    if (sdata[tid + ( n ) ] > sdata[tid]) { \
      sdata[tid] = sdata[tid + ( n ) ]; \
      indices[tid] = indices[tid + ( n ) ]; \
    } \
  }

__global__ void gMaxElement(mblas::MatrixWrapper<NthOut> out,
                            const mblas::MatrixWrapper<float> probsWrap,
                            const mblas::MatrixWrapper<uint> batchPositionWrap,
                            uint numBatches)
{
  extern __shared__ float sdata[];
  __shared__ uint indices[SHARED_SIZE];

  uint tid = threadIdx.x;

  for (uint batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    uint begin = batchPositionWrap[batchIdx];
    uint end = batchPositionWrap[batchIdx + 1];

    uint i = begin + blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = -3.40282e+38f;

    if (i < end) {
      sdata[tid] = probsWrap[i];
      indices[tid] = i;
    }

    if (i + blockDim.x < end) {
      float a = probsWrap[i];
      float b = probsWrap[i + blockDim.x];
      if (a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + 2 * gridDim.x * blockDim.x < end) {
      i += 2 * gridDim.x * blockDim.x;

      float a = probsWrap[i];
      if (a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < end) {
        float b = probsWrap[i + blockDim.x];
        if (b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (uint s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < end) {
        if (sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP(32, end);
    UNROLL_MAXARG_LOOP(16, end);
    UNROLL_MAXARG_LOOP(8, end);
    UNROLL_MAXARG_LOOP(4, end);
    UNROLL_MAXARG_LOOP(2, end);
    UNROLL_MAXARG_LOOP(1, end);

    if (tid == 0) {
      out[blockIdx.x + batchIdx * gridDim.x] = {indices[0], sdata[0]};
    }
    __syncthreads();
  }
}

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOut> out,
                                  mblas::MatrixWrapper<float> probsWrap,
                                  mblas::MatrixWrapper<uint> batchPositionWrap,
                                  mblas::MatrixWrapper<NthOut> resNewWrap,
                                  mblas::MatrixWrapper<uint> cumBeamSizesWrap,
                                  uint numBlocks)
{
  extern __shared__ float sdata[];
  __shared__ uint indices[SHARED_SIZE];
  __shared__ float bestBinCost;
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

    sdata[tid] = -3.40282e+38f;

    if (i < num_bins) {
      sdata[tid] = out[batchIdx * numBlocks + i].score;
      indices[tid] = i;
    }

    if (i + blockDim.x < num_bins) {
      float a = out[batchIdx * numBlocks + i].score;
      float b = out[batchIdx * numBlocks + i + blockDim.x].score;
      if (a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + 2 * blockDim.x < num_bins) {
      i += 2 * blockDim.x;

      float a = out[batchIdx * numBlocks + i].score;
      if (a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < num_bins) {
        float b = out[batchIdx * numBlocks + i + blockDim.x].score;
        if (b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (uint s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < num_bins) {
        if (sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP(32, num_bins);
    UNROLL_MAXARG_LOOP(16, num_bins);
    UNROLL_MAXARG_LOOP(8, num_bins);
    UNROLL_MAXARG_LOOP(4, num_bins);
    UNROLL_MAXARG_LOOP(2, num_bins);
    UNROLL_MAXARG_LOOP(1, num_bins);

    if (tid == 0) {
      bestBinCost = sdata[0];
      bestBinCostIdx = batchIdx * numBlocks + indices[0];

      probsWrap[ out[bestBinCostIdx].ind ] = -3.40282e+38f;

      resNewWrap[pos].ind = out[bestBinCostIdx].ind;
      resNewWrap[pos].score = bestBinCost;
    }

    __syncthreads();

    i = batchPositionWrap[batchIdx] + (bestBinCostIdx - batchIdx * numBlocks) * (blockDim.x * 2) + tid;
    const uint dist = num_bins * 2 * blockDim.x;

    sdata[tid] = -3.40282e+38f;

    if (i < batchPositionWrap[batchIdx + 1]) {
      sdata[tid] = probsWrap[i];
      indices[tid] = i;
    }

    if (i + blockDim.x < batchPositionWrap[batchIdx + 1]) {
      float a = probsWrap[i];
      float b = probsWrap[i+blockDim.x];
      if (a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + dist < batchPositionWrap[batchIdx + 1]) {
      i += dist;

      float a = probsWrap[i];
      if (a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < batchPositionWrap[batchIdx + 1]) {
        float b = probsWrap[i + blockDim.x];
        if (b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (uint s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < batchPositionWrap[batchIdx + 1]) {
        if (sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP(32, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(16, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(8, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(4, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(2, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(1, batchPositionWrap[batchIdx + 1]);

    if (tid == 0) {
      out[bestBinCostIdx] = {indices[0], sdata[0]};
    }
    __syncthreads();
  }
}

__global__ void gGetValueByKey(mblas::MatrixWrapper<float> out,
                              const   mblas::MatrixWrapper<float> in,
                              uint* indices, uint n)
{
  uint tid = threadIdx.x  + blockDim.x * blockIdx.x;
  if (tid < n) {
    uint index = indices[tid];
    out[tid] = in[index];
  }
}

/////////////////////////////////////////////////////////////////////////////////

#define UNROLL_MAXARG_LOOP_HALF( n, max ) \
  if (tid < (n) && tid + (n) < ( max ) ) { \
    if (sdataHalf[tid + ( n ) ] > sdataHalf[tid]) { \
      sdataHalf[tid] = sdataHalf[tid + ( n ) ]; \
      indices[tid] = indices[tid + ( n ) ]; \
    } \
  }

__global__ void gMaxElement(mblas::MatrixWrapper<NthOutHalf> out,
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

    sdataHalf[tid] = -3.40282e+38f;

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

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOutHalf> out,
                                  mblas::MatrixWrapper<half> probsWrap,
                                  mblas::MatrixWrapper<uint> batchPositionWrap,
                                  mblas::MatrixWrapper<NthOutHalf> resNewWrap,
                                  mblas::MatrixWrapper<uint> cumBeamSizesWrap,
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

    sdataHalf[tid] = -3.40282e+38f;

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

      probsWrap[ out[bestBinCostIdx].ind ] = -3.40282e+38f;

      resNewWrap[pos].ind = out[bestBinCostIdx].ind;
      resNewWrap[pos].score = bestBinCostHalf;
    }

    __syncthreads();

    i = batchPositionWrap[batchIdx] + (bestBinCostIdx - batchIdx * numBlocks) * (blockDim.x * 2) + tid;
    const uint dist = num_bins * 2 * blockDim.x;

    sdataHalf[tid] = -3.40282e+38f;

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

