// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define SECTION_SIZE 2 * BLOCK_SIZE

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *X, float *Y, int len, float *a) {
  __shared__ float XY[SECTION_SIZE];
  int i = (2 * blockIdx.x * blockDim.x)+ threadIdx.x;

  if (i < len)  {
    XY[threadIdx.x] = X[i];
  } else {
    XY[threadIdx.x] = 0.0;
  }

  if (i + blockDim.x < len) {
    XY[threadIdx.x+blockDim.x] = X[i+blockDim.x];
  } else {
    XY[threadIdx.x+blockDim.x] = 0.0;
  }

  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < (SECTION_SIZE)) {
      XY[index] += XY[index - stride];
    }
  }
  for (int stride = (SECTION_SIZE) / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index + stride < (SECTION_SIZE)) {
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads();

  if (i < len) {
    Y[i] = XY[threadIdx.x];
  }
  if (i + blockDim.x < len) {
    Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
  }
  __syncthreads();

  if(threadIdx.x == blockDim.x - 1) {
    a[blockIdx.x] = XY[SECTION_SIZE - 1];
  } 
}

__global__ void add(float *X, float *Y, int len) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(blockIdx.x > 0 && i < len) {
    
    Y[i] += X[blockIdx.x - 1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  int numBlocks = ceil(float(numElements) / (SECTION_SIZE));

  float *hostAuxiliaryArray = (float *)malloc(numBlocks * sizeof(float));
  float *deviceAuxiliaryArray;
  wbCheck(cudaMalloc((void **) &deviceAuxiliaryArray, numBlocks * sizeof(float)));

  dim3 DimGrid(numBlocks, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements, deviceAuxiliaryArray);

  float *deviceAuxiliaryArrayScan;
  wbCheck(cudaMalloc((void **) &deviceAuxiliaryArrayScan, numBlocks * sizeof(float)));

  float *hostAuxiliaryArrayScan = (float *)malloc(numBlocks * sizeof(float)) ;
  scan<<<DimGrid, DimBlock>>>(deviceAuxiliaryArray, deviceAuxiliaryArrayScan, numBlocks, deviceAuxiliaryArray);

  dim3 DimGridAdd(ceil(float(numElements) / (SECTION_SIZE)), 1, 1);
  dim3 DimBlockAdd((SECTION_SIZE), 1, 1);

  add<<<DimGridAdd, DimBlockAdd>>>(deviceAuxiliaryArrayScan, deviceOutput, numElements);

  wbTime_start(Compute, "Performing CUDA computation");

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxiliaryArray);
  cudaFree(deviceAuxiliaryArrayScan);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}


// http://s3.amazonaws.com/files.rai-project.com/userdata/build-635a253cb1b11b68c5a008d3.tar.gz