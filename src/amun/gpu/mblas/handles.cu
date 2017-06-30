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
  CreateHandle(handle_, CudaStreamHandler::GetStream());
  CreateHandle(handleEnc_, CudaStreamHandler::GetEncoderStream());
}

CublasHandler::~CublasHandler() {
  cublasDestroy(handle_);
}

void CublasHandler::CreateHandle(cublasHandle_t &handle, const cudaStream_t &stream) const
{
  cublasStatus_t stat;
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
  printf ("cublasCreate initialization failed\n");
  abort();
  }

  stat = cublasSetStream(handle, stream);
  if (stat != CUBLAS_STATUS_SUCCESS) {
  printf ("cublasSetStream initialization failed\n");
  abort();
  }

}


}
}
}
