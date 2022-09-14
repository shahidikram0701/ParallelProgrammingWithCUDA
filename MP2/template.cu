
#include <wb.h>

// #define PRINTOUTPUT

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLOCK_SIZE 32

__global__
void matMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if((r < numCRows) && (c < numCColumns)) {
      float sum = 0.0;
      for(int i = 0; i < numAColumns; ++i) {
        sum += (A[r * numAColumns + i] * B[i * numBColumns + c]);
      }
      C[r * numCColumns + c] = sum;
    }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  // ____________________Set numCRows and numCColumns______________________
 
  numCRows = numARows;
  numCColumns = numBColumns;

  // _____________________Allocating the hostC matrix_______________________
  
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
   
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  

  // ___________________Allocating GPU memory here_________________________

  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  // _________________Copying memory to the GPU here______________________
  
  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //_________Initializing the grid and block dimensions here______________
  wbTime_start(Compute, "Performing CUDA computation");
  int blockSizeX = ceil((1.0 * numCColumns) / BLOCK_SIZE);
  int blockSizeY = ceil((1.0 * numCRows) / BLOCK_SIZE);
  dim3 DimGrid(blockSizeX, blockSizeY, 1);
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  matMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  //____________Copying the GPU memory back to the CPU here_________________

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  #ifdef PRINTOUTPUT
  for(int i = 0; i < numCRows; ++i) {
    for(int j = 0; j < numCColumns; ++j) {
      printf("%f ", hostC[i * numCColumns + j]);
    }
  }
  #endif

  //_________________Freeing the GPU memory here_____________________________
  
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
