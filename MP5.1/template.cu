// MP 5.1 Reduction
// Given a list of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>

#define BLOCK_SIZE 256 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len) {
  __shared__ float buffer[2 * BLOCK_SIZE];

  int tx = threadIdx.x;
  int i = (blockIdx.x * blockDim.x) * 2 + tx;

  buffer[tx] = input[i];
  if((i + BLOCK_SIZE) < len) {
     buffer[tx + BLOCK_SIZE] = input[i + BLOCK_SIZE];
  } else {
    buffer[tx + BLOCK_SIZE] = 0.0;
  }
 

  for(int stride = blockDim.x; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tx < stride)
      buffer[tx] = buffer[tx] + buffer[tx + stride];
  }

  output[blockIdx.x] = buffer[0];
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = (numInputElements - 1) / (BLOCK_SIZE << 1) + 1;
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //___________________Allocating GPU memory here______________________________

  cudaMalloc((void **) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void **) &deviceOutput, numOutputElements * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  
  //___________________Copying memory to the GPU here_________________________

  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //________________Initializing the grid and block dimensions here___________

  dim3 DimGrid(ceil(float(numInputElements) / (BLOCK_SIZE << 1)), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");

  total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements); 

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");

  //_____________Copy the GPU memory back to the CPU here_________________

  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /***********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input.
   * For simplicity, we do not require that for this lab!
   ***********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");

  //_____________________Freeing the GPU memory here_______________________

  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}


// http://s3.amazonaws.com/files.rai-project.com/userdata/build-6350d9f9b1b11b75c274158b.tar.gz