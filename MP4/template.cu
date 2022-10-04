#include <wb.h>

#define TILE_SIZE 4
#define MASK_WIDTH 3

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
__constant__ 
float mask[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ 
void conv3d(float *input, float *output, const int plane_size,
                       const int row_size, const int column_size) {
    
    __shared__ float tile[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE + MASK_WIDTH - 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int output_col = bx * TILE_SIZE + tx;
    int output_row = by * TILE_SIZE + ty;
    int output_plane = bz * TILE_SIZE + tz;

    int input_col = output_col - (MASK_WIDTH / 2);
    int input_row = output_row - (MASK_WIDTH / 2);
    int input_plane = output_plane - (MASK_WIDTH / 2);

    if( input_col >= 0 && input_col < column_size && 
        input_row >= 0 && input_row < row_size && 
        input_plane >= 0 && input_plane < plane_size ) {
          tile[tz][ty][tx] = input[(input_plane * column_size * row_size) + (input_row * column_size) + input_col];
    } else {
          tile[tz][ty][tx] = 0.0;
    }

    __syncthreads();
    // if(bx == 0 && by == 0 && bz == 0 && tx == 0 && ty == 0 && tz == 0) {
    //   for(int k = 0; k < TILE_SIZE + MASK_WIDTH - 1; ++k) {
    //     for(int i = 0; i < TILE_SIZE + MASK_WIDTH - 1; ++i) {
    //       for(int j = 0; j < TILE_SIZE + MASK_WIDTH - 1; ++j) {
    //         printf("%f ", tile[k][i][j]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }
    float sum = 0.0;
    if(ty < TILE_SIZE && tx < TILE_SIZE && tz < TILE_SIZE) {
      for(int k = 0; k < MASK_WIDTH; k++) {
        for(int i = 0; i < MASK_WIDTH; i++) {
          for(int j = 0; j < MASK_WIDTH; ++j) {
            // so the mask is aligned with the top left of the tile
            // this computes first element of the output tile
            // in order to move the mask, I am adding the thread offset to the tile indices
            sum += (mask[k][i][j] * tile[tz + k][ty + i][tx + j]);
          }
        }
      }
      if(output_col < column_size && output_row < row_size && output_plane < plane_size) {
      // printf("[%d][%d][%d]\n", output_plane, output_row, output_col);
        output[output_plane * column_size * row_size + output_row * column_size + output_col] = sum;
      }
    }

    

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int plane_size;
  int row_size;
  int column_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  plane_size = hostInput[0];
  row_size = hostInput[1];
  column_size = hostInput[2];
  wbLog(TRACE, "The input size is ", plane_size, "x", row_size, "x", column_size);
  assert(plane_size * row_size * column_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  int numElements = (inputLength - 3);
  
  //_____________________ Allocating GPU memory _____________________________
  wbTime_start(GPU, "Doing GPU memory allocation");
  
  cudaMalloc((void **) &deviceInput, numElements * sizeof(float));
  cudaMalloc((void **) &deviceOutput, numElements * sizeof(float));
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  // ____________________ Copy input and kernel to GPU here __________________
  wbTime_start(Copy, "Copying data to the GPU");

  cudaMemcpy(deviceInput, hostInput + 3, numElements * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  
  dim3 DimBlock(
    TILE_SIZE + MASK_WIDTH - 1, 
    TILE_SIZE + MASK_WIDTH - 1, 
    TILE_SIZE + MASK_WIDTH - 1
  );

  dim3 DimGrid(
    ceil(float(column_size) / TILE_SIZE),
    ceil(float(row_size) / TILE_SIZE),
    ceil(float(plane_size) / TILE_SIZE)
  );

  // Copying the mask to shared memory
  cudaMemcpyToSymbol(mask, hostKernel, kernelLength * sizeof(float));
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, plane_size, row_size, column_size);

  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  // ______________ Copy the device memory back to the host ___________________
  wbTime_start(Copy, "Copying data from the GPU");

  cudaMemcpy(hostOutput + 3, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = plane_size;
  hostOutput[1] = row_size;
  hostOutput[2] = column_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
