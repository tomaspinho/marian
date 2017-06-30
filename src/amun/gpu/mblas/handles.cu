#include "handles.h"
#include "gpu/types-gpu.h"

namespace amunmt {
namespace GPU {
namespace mblas {

CudaStreamHandler::CudaStreamHandler()
{
  HANDLE_ERROR( cudaStreamCreate(&stream_));
  HANDLE_ERROR( cudaStreamCreate(&streamEnc_));
  // cudaStreamCreateWithFlags(stream_.get(), cudaStreamNonBlocking);
}

CudaStreamHandler::~CudaStreamHandler()
{
  HANDLE_ERROR(cudaStreamDestroy(stream_));
  HANDLE_ERROR(cudaStreamDestroy(streamEnc_));
}

///////////////////////////////////////////////////////////////////////
CublasHandler::CublasHandler()
{
  cublasStatus_t stat;
  stat = cublasCreate(&handle_);
  if (stat != CUBLAS_STATUS_SUCCESS) {
  printf ("cublasCreate initialization failed\n");
  abort();
  }

  stat = cublasSetStream(handle_, CudaStreamHandler::GetStream());
  if (stat != CUBLAS_STATUS_SUCCESS) {
  printf ("cublasSetStream initialization failed\n");
  abort();
  }
}

CublasHandler::~CublasHandler() {
  cublasDestroy(handle_);
}


}
}
}
