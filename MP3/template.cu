
#include <wb.h>

#define BLOCK_SIZE 32
#define TILE_WIDTH BLOCK_SIZE
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

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

  float answer = 0;
  for(int phase = 0; phase < ceil(float(numAColumns) / TILE_WIDTH); ++phase) {
    
    // sharedA will contain A[phase * TILE_WIDTH + threadIdx.x][row]
    if((row < numARows) && (phase * TILE_WIDTH + threadIdx.x) < numAColumns) {
      sharedA[threadIdx.y][threadIdx.x] = A[(row * numAColumns) + ((phase * TILE_WIDTH)+ threadIdx.x)];
    } else {
      sharedA[threadIdx.y][threadIdx.x] = 0.0; 
    }

  // sharedB will contain B[col][phase * TILE_WIDTH + threadIdx.y]
    if (col < numBColumns && (((phase * TILE_WIDTH) + threadIdx.y) < numBRows)) {
      sharedB[threadIdx.y][threadIdx.x] = B[(((phase * TILE_WIDTH) + threadIdx.y) * numBColumns) + col];
    } else {
      sharedB[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    for(int i = 0; i < TILE_WIDTH; ++i) {
      answer += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
    }

    __syncthreads();

    if(row < numCRows && col < numCColumns) {
      C[row * numCColumns + col] = answer;
    } 
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

  wbTime_start(GPU, "Copying input memory to the GPU.");

  // _________________Copying memory to the GPU here______________________

  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //_________Initializing the grid and block dimensions here______________

  int blockSizeX = ceil((1.0 * numCColumns) / BLOCK_SIZE);
  int blockSizeY = ceil((1.0 * numCRows) / BLOCK_SIZE);
  dim3 DimGrid(blockSizeX, blockSizeY, 1);
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  
  cudaMemcpy(hostC, deviceC, numCColumns * numCRows * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");


  #ifdef PRINTOUTPUT
  for(int i = 0; i < numCRows; ++i) {
    for(int j = 0; j < numCColumns; ++j) {
      printf("%f ", hostC[i * numCColumns + j]);
    }
    printf("\n");
  }
  #endif

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
