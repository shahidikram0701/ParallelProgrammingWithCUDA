#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8
#define TILE_WIDTH2 8
#define TILE_WIDTH3 32
#define TILE_WIDTH4 4
// #define NUM_STREAMS 10 //Uncomment this line to run the stream convolution

__constant__ 
float mask[5000];

__global__ void unroll(int b, int C, const int H, const int W, const int K, const float* X, float* X_unroll) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; 
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    int H_unroll = C * K * K;

    #define in_4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define out_2d(i1, i0) X_unroll[(i1) * (W_unroll) + i0]

    if (t < C * W_unroll) {
        // printf("t: %d\n", t);
        int c = t / W_unroll;
        int s = t % W_unroll; 
        int h_out = s / W_out;
        int w_out = s % W_out;
        int h_unroll = h_out * W_out + w_out; 
        int w_base = c * K * K;
        // printf("w_base: %d\n", w_base);
        for(int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                int w_unroll = w_base + p * K + q;
                int input = in_4d(b, c, h_out + p, w_out + q);
                // printf("t: %d; h_unroll: %d; w_unroll: %d; c: %d; h_out + p: %d; w_out + q: %d; input: %d\n", t, h_unroll, w_unroll, c, h_out + p, w_out + q, input);
                out_2d(w_unroll, h_unroll) = input;
            }
        }
    }
    
    #undef in_4d 
    #undef out_2d
}


__global__ void unroll_mask(int C, int K, int map_out, float* mask, float* map_unrolled) {
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int startIdx = tx * K * K;

    int map_number = tx / C;
    int channel = tx % C;
    if(tx < (map_out * C)) {
        for(int i = 0; i < K; ++i) {
            for(int j = 0; j < K; ++j) {
                map_unrolled[startIdx + i * K + j] = mask_4d(map_number, channel, i, j);
            }
        }
    }

    #undef mask_4d
}


__global__ void conv_forward_kernel(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    
    int maskNum = blockIdx.x;
    int Width_size = ceil(float(Width_out) / TILE_WIDTH);

    int output_i = (blockIdx.y / Width_size) * TILE_WIDTH + threadIdx.y;
    int output_j = (blockIdx.y % Width_size) * TILE_WIDTH + threadIdx.x;

    int b = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (b < Batch && output_i < Height_out && output_j < Width_out) {
        float acc = 0.0;
        for(int c = 0; c < Channel; ++c) {
            for (int mask_i = 0; mask_i < K; ++mask_i) {
                for (int mask_j = 0; mask_j < K; ++mask_j) {
                    acc += (in_4d(b, c, output_i + mask_i, output_j + mask_j) * mask_4d(maskNum, c, mask_i, mask_j));
                }    
            }
        }
        out_4d(b, maskNum, output_i, output_j) = acc;
    } 


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__global__ void matrixMultiply(float *B,  
                               float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
//   __shared__ float sharedA[TILE_WIDTH3][TILE_WIDTH3];
  __shared__ float sharedB[TILE_WIDTH3][TILE_WIDTH3];

  int col = blockIdx.x * TILE_WIDTH3 + threadIdx.x;
  int row = blockIdx.y * TILE_WIDTH3 + threadIdx.y;

  float answer = 0;
  for(int phase = 0; phase < ceil(float(numAColumns) / TILE_WIDTH3); ++phase) {
    
    // sharedA will contain A[phase * TILE_WIDTH + threadIdx.x][row]
    // if((row < numARows) && (phase * TILE_WIDTH3 + threadIdx.x) < numAColumns) {
    //   sharedA[threadIdx.y][threadIdx.x] = A[(row * numAColumns) + ((phase * TILE_WIDTH3)+ threadIdx.x)];
    // } else {
    //   sharedA[threadIdx.y][threadIdx.x] = 0.0; 
    // }

  // sharedB will contain B[col][phase * TILE_WIDTH + threadIdx.y]
    if (col < numBColumns && (((phase * TILE_WIDTH3) + threadIdx.y) < numBRows)) {
      sharedB[threadIdx.y][threadIdx.x] = B[(((phase * TILE_WIDTH3) + threadIdx.y) * numBColumns) + col];
    } else {
      sharedB[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    for(int i = 0; i < TILE_WIDTH3; ++i) {
      answer += mask[(row * numAColumns) + ((phase * TILE_WIDTH3) + i)] * sharedB[i][threadIdx.x];
    }

    __syncthreads();

    if(row < numCRows && col < numCColumns) {
      C[row * numCColumns + col] = answer;
    } 
  }
}

__global__ void conv_forward_kernel_tiled(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    int b = blockDim.z * blockIdx.z + threadIdx.z;

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int Width_size = ceil(float(Width_out) / TILE_WIDTH2);

    int sharedMemoryWidth = TILE_WIDTH2 + K - 1;
    extern __shared__ float shmem[];
    float* shmem_input = &shmem[threadIdx.z * sharedMemoryWidth * sharedMemoryWidth];

    int maskNum = blockIdx.x;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int by = (blockIdx.y / Width_size) * TILE_WIDTH2;
    int bx = (blockIdx.y % Width_size) * TILE_WIDTH2;

    int output_i = by + ty;
    int output_j = bx + tx;

    float acc = 0.0;
    if(b < Batch) {
        for(int c = 0; c < Channel; ++c) {
            for (int i = output_i; i < (by + sharedMemoryWidth); i += TILE_WIDTH2) { 
                for (int j = output_j; j < (bx + sharedMemoryWidth); j += TILE_WIDTH2) {
                    shmem_input[((i - by) * sharedMemoryWidth + (j - bx))] = in_4d(b, c, i, j); 

                }
            }

            __syncthreads();

            for (int mask_i = 0; mask_i < K; ++mask_i) {
                for (int mask_j = 0; mask_j < K; ++mask_j) {
                    acc += shmem_input[(output_i - by + mask_i) * sharedMemoryWidth + (output_j - bx + mask_j)] * mask_4d(maskNum, c, mask_i, mask_j);
                }    
            }

            __syncthreads();
        }
    }
    if (b < Batch && output_i < Height_out && output_j < Width_out) {
        out_4d(b, maskNum, output_i, output_j) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d

}


__global__ void conv_forward_kernel_tiled_tree_reduction(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int b = (blockIdx.z * blockDim.z + threadIdx.z) / Channel;

    int Width_size = ceil(float(Width_out) / TILE_WIDTH4);

    int sharedMemoryWidth = (TILE_WIDTH4 + K - 1);
    extern __shared__ float shmem[];

    float* shmem_input =  &shmem[(threadIdx.z / Channel) * Channel * sharedMemoryWidth * sharedMemoryWidth];
    float* shmem_channel_data = &shmem_input[Channel * sharedMemoryWidth * sharedMemoryWidth];

    #define sc_3d(i2, i1, i0) shmem_channel_data[(i2) * (TILE_WIDTH4 * TILE_WIDTH4) + (i1) * (TILE_WIDTH4) + i0]

    int maskNum = blockIdx.x;
    
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int by = (blockIdx.y / Width_size) * TILE_WIDTH4;
    int bx = (blockIdx.y % Width_size) * TILE_WIDTH4;
    int c = tz % Channel;

    int output_i = by + ty;
    int output_j = bx + tx;

    float acc = 0.0;
    
    if(c < Channel) {
        for (int i = output_i; i < (by + sharedMemoryWidth); i += TILE_WIDTH4) { 
            for (int j = output_j; j < (bx + sharedMemoryWidth); j += TILE_WIDTH4) {
                shmem_input[((c * sharedMemoryWidth * sharedMemoryWidth) + ((i - by) * sharedMemoryWidth) + (j - bx))] = in_4d(b, c, i, j); 
            }
        }
        __syncthreads();

        for (int mask_i = 0; mask_i < K; ++mask_i) {
            for (int mask_j = 0; mask_j < K; ++mask_j) {
                acc += shmem_input[((c * sharedMemoryWidth * sharedMemoryWidth) + (output_i - by + mask_i) * sharedMemoryWidth + (output_j - bx + mask_j))] * mask_4d(maskNum, c, mask_i, mask_j);
            }
        }
        sc_3d(c, ty, tx) = acc;

        // do a reduction on shmem_channel_data
        for(int stride = (Channel / 2); stride > 0; stride /= 2) {
            __syncthreads();
            if (c < stride) {
                sc_3d(c, ty, tx) += sc_3d(c + stride, ty, tx);
            }
        }
    
        if (b < Batch && output_i < Height_out && output_j < Width_out) {
            out_4d(b, maskNum, output_i, output_j) = sc_3d(0, ty, tx);
        }

        // if (b < Batch && output_i < Height_out && output_j < Width_out) {
        //     atomicAdd(&out_4d(b, maskNum, output_i, output_j), shmem_channel_data[c]);
        // }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef sc_3d
}

__global__ void conv_forward_kernel_tiled_atomics(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int b = (blockIdx.z * blockDim.z + threadIdx.z) / Channel;

    int Width_size = ceil(float(Width_out) / TILE_WIDTH4);

    int sharedMemoryWidth = (TILE_WIDTH4 + K - 1);
    extern __shared__ float shmem[];

    float* shmem_input =  &shmem[(threadIdx.z / Channel) * Channel * sharedMemoryWidth * sharedMemoryWidth];
    // float* shmem_channel_data = &shmem_input[Channel * sharedMemoryWidth * sharedMemoryWidth];

    // #define sc_3d(i2, i1, i0) shmem_channel_data[(i2) * (TILE_WIDTH4 * TILE_WIDTH4) + (i1) * (TILE_WIDTH4) + i0]

    int maskNum = blockIdx.x;
    
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int by = (blockIdx.y / Width_size) * TILE_WIDTH4;
    int bx = (blockIdx.y % Width_size) * TILE_WIDTH4;
    int c = tz % Channel;

    int output_i = by + ty;
    int output_j = bx + tx;

    float acc = 0.0;
    
    if(b < Batch && c < Channel) {
        for (int i = output_i; i < (by + sharedMemoryWidth); i += TILE_WIDTH4) { 
            for (int j = output_j; j < (bx + sharedMemoryWidth); j += TILE_WIDTH4) {
                shmem_input[((c * sharedMemoryWidth * sharedMemoryWidth) + ((i - by) * sharedMemoryWidth) + (j - bx))] = in_4d(b, c, i, j); 
            }
        }
        __syncthreads();

        for (int mask_i = 0; mask_i < K; ++mask_i) {
            for (int mask_j = 0; mask_j < K; ++mask_j) {
                acc += shmem_input[((c * sharedMemoryWidth * sharedMemoryWidth) + (output_i - by + mask_i) * sharedMemoryWidth + (output_j - bx + mask_j))] * mask_4d(maskNum, c, mask_i, mask_j);
            }
        }
        // sc_3d(c, ty, tx) = acc;

        // // do a reduction on shmem_channel_data
        // for(int stride = (Channel / 2); stride > 0; stride /= 2) {
        //     __syncthreads();
        //     if (c < stride) {
        //         sc_3d(c, ty, tx) += sc_3d(c + stride, ty, tx);
        //     }
        // }
    
        // if (b < Batch && output_i < Height_out && output_j < Width_out) {
        //     out_4d(b, maskNum, output_i, output_j) = sc_3d(0, ty, tx);
        // }

        if (b < Batch && output_i < Height_out && output_j < Width_out) {
            atomicAdd(&out_4d(b, maskNum, output_i, output_j), acc);
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    // #undef sc_3d
}


__global__ void fusedUnrollAndMatMul(int C, const int H, const int W, const int K, const float* X, float* output, int Map_out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; 
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    int H_unroll = C * K * K;

    int b = blockIdx.z * blockDim.z + threadIdx.z;

    float* op = output + (b * (W_out * H_out) * Map_out);

    #define in_4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // #define out_2d(i1, i0) X_unroll[(i1) * (W_unroll) + i0]

    if (t < C * W_unroll) {
        // printf("t: %d\n", t);
        int c = t / W_unroll;
        int s = t % W_unroll; 
        int h_out = s / W_out;
        int w_out = s % W_out;
        int h_unroll = h_out * W_out + w_out; // column number
        int w_base = c * K * K;
        // printf("w_base: %d\n", w_base);
        for(int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                int w_unroll = w_base + p * K + q;
                int input = in_4d(b, c, h_out + p, w_out + q);
                // printf("t: %d; h_unroll: %d; w_unroll: %d; c: %d; h_out + p: %d; w_out + q: %d; input: %d\n", t, h_unroll, w_unroll, c, h_out + p, w_out + q, input);
                
                for(int m = 0; m < Map_out; ++m) {
                //    output[m * H_out * W_out + h_unroll] += mask[m * C * K * K + w_unroll] * input
                   atomicAdd(&op[m * H_out * W_out + h_unroll], mask[m * C * K * K + w_unroll] * input);
                }  
            }
        }
    }
    
    #undef in_4d   
}


__global__ void restrict_loop_unroll_kernel(float* __restrict__ output, const float* __restrict__ input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    
    int maskNum = blockIdx.x;
    int Width_size = ceil(float(Width_out) / TILE_WIDTH);

    int output_i = (blockIdx.y / Width_size) * TILE_WIDTH + threadIdx.y;
    int output_j = (blockIdx.y % Width_size) * TILE_WIDTH + threadIdx.x;

    int b = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (b < Batch && output_i < Height_out && output_j < Width_out) {
        float acc = 0.0;
        for(int c = 0; c < Channel; ++c) {
            #pragma unroll 7 // telling the compiler to unroll the below for loop K times where K is 7
            for (int mask_i = 0; mask_i < K; ++mask_i) {
                #pragma unroll 7 //telling the compiler to unroll the below loop K times
                for (int mask_j = 0; mask_j < K; ++mask_j) {
                    acc += (in_4d(b, c, output_i + mask_i, output_j + mask_j) * mask_4d(maskNum, c, mask_i, mask_j));
                }    
            }
        }
        out_4d(b, maskNum, output_i, output_j) = acc;
    } 


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    #ifdef NUM_STREAMS
    conv_forward_gpu_prolog_stream(host_output, host_input, host_mask, device_output_ptr, device_input_ptr, device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);

    return;
    #endif

    // Allocate memory and copy over the relevant data structures to the GPU

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void **)device_input_ptr, Batch * ((Width * Height) * Channel) * sizeof(float));

    cudaMalloc((void **)device_output_ptr, Batch * ((Width_out * Height_out) * Map_out) * sizeof(float));

    cudaMemcpyToSymbol(mask, host_mask, Channel * K * K * Map_out * sizeof(float));

    // cudaMalloc((void **)device_mask_ptr, Map_out * ((K * K) * Channel) * sizeof(float));

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    cudaMemcpy(*device_input_ptr, 
                host_input, 
                Batch * ((Width * Height) * Channel) * sizeof(float), 
                cudaMemcpyHostToDevice);

    // cudaMemcpy(*device_mask_ptr, 
    //             host_mask, 
    //             Map_out * ((K * K) * Channel) * sizeof(float), 
    //             cudaMemcpyHostToDevice);
}


__host__ void conv_forward_gpu_prolog_stream(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    #ifdef NUM_STREAMS
   cout << "\n\nLaunching conv kernel with streams\n\n" << endl;
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    cudaMalloc((void**)device_output_ptr, Batch * Map_out * H_out * W_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpyToSymbol(mask, host_mask, Channel * K * K * Map_out * sizeof(float));

    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }
    
    int inputBatchSize = (Batch * Channel * Height * Width) / NUM_STREAMS;
    int outputBatchSize = (Batch * Map_out * H_out * W_out) / NUM_STREAMS;

    const int W_size = ceil(float(W_out) / TILE_WIDTH2); 
    const int H_size = ceil(float(H_out) / TILE_WIDTH2);
    const int Y = H_size * W_size;
    const int Z = ceil(float(Batch) / NUM_STREAMS);

    // dim3 DimBlock_Unroll(32, 1, 10);
    // dim3 DimGrid_Unroll(ceil((Channel * H_out * W_out) / 32.0), 1, Z / 10);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, Y, Z);

    // dim3 blockDim2(TILE_WIDTH2, TILE_WIDTH2, TILE_WIDTH2);
    // dim3 gridDim2(Map_out, Y, ceil(float(Z) / TILE_WIDTH2));
    

    float* host_output_temp = (float*)host_output;
    size_t shmem_size = sizeof(float) * (TILE_WIDTH2 * (TILE_WIDTH2 + (K - 1)) * (TILE_WIDTH2 + (K - 1)));

    for (int i = 0; i < NUM_STREAMS; i++){
        int inputPart = inputBatchSize * i;
        int outputPart = outputBatchSize * i;

        cudaMemcpyAsync((*device_input_ptr) + inputPart, host_input + inputPart, inputBatchSize * sizeof(float), cudaMemcpyHostToDevice, stream[i]);

        // fusedUnrollAndMatMul<<<DimGrid_Unroll, DimBlock_Unroll, 0, stream[i]>>>(Channel, Height, Width, K, (*device_input_ptr) + inputPart, (*device_output_ptr) + outputPart, Map_out);

        conv_forward_kernel<<<gridDim, blockDim, 0, stream[i]>>>((*device_output_ptr) + outputPart, (*device_input_ptr) + inputPart, Batch, Map_out, Channel, Height, Width, K);
    
        // conv_forward_kernel_tiled<<<gridDim2, blockDim2, shmem_size>>>((*device_output_ptr) + outputPart, (*device_input_ptr) + inputPart, Batch, Map_out, Channel, Height, Width, K);

        cudaMemcpyAsync(host_output_temp + outputPart, (*device_output_ptr) + outputPart, outputBatchSize * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(stream[i]);

    // Free device memory
    cudaFree(device_input_ptr);
    cudaFree(device_output_ptr);
    cudaFree(device_mask_ptr);
    #endif
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{   
    #ifdef NUM_STREAMS
        return;
    #endif

    printf("BatchSize: %d\n", Batch);
    printf("Map_out: %d\n", Map_out);
    printf("Channel: %d\n", Channel);
    printf("Height: %d\n", Height);
    printf("Width: %d\n", Width);
    printf("K: %d\n", K);
    
    // Baseline with constant memory
    // launchBaselineKernel(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);

    // Tiled Convolution
    // launchTiledKernel(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);


    // Input Channel Reduction - Tree
    // launchTiledKernelChannelTreeReduction(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);

    // Input Channel Reduction - Atomics
    // launchTiledKernelChannelReductionAtomics(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);

    // Unrolled Matrix Multiplication - Poor performance - need to optimise
    // launchUnrolledMatrixMultiplicationKernel(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);

    // Fused Unrolling and Matrix Multiplication - Slightly less performant - Need to optimise further
    // launchFusedUnrolledMatrixMultiplicationKernel(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);

    // Loop unrolling kernel
    launchRestrictLoopUnrollKernel(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);


}


__host__ void launchBaselineKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    cout << "\n\nLaunching Baseline Kernel with weights in the constant memory \n\n" << endl;
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int W_size = ceil(float(Width_out) / TILE_WIDTH); 
    const int H_size = ceil(float(Height_out) / TILE_WIDTH);
    const int Y = H_size * W_size;
    const int Z = ceil(float(Batch) / TILE_WIDTH);


    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(Map_out, Y, Z);

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void launchTiledKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    cout << "\n\nLaunching Tiled Kernel\n\n" << endl;
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    size_t shmem_size = sizeof(float) * (TILE_WIDTH2 * (TILE_WIDTH2 + (K - 1)) * (TILE_WIDTH2 + (K - 1)));

    const int W_size2 = ceil(float(Width_out) / TILE_WIDTH2); 
    const int H_size2 = ceil(float(Height_out) / TILE_WIDTH2);
    const int Y2 = H_size2 * W_size2;
    


    dim3 blockDim2(TILE_WIDTH2, TILE_WIDTH2, TILE_WIDTH2);
    dim3 gridDim2(Map_out, Y2, ceil(float(Batch) / TILE_WIDTH2));
    conv_forward_kernel_tiled<<<gridDim2, blockDim2, shmem_size>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void launchTiledKernelChannelTreeReduction(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    cout << "\n\nLaunching Tiled Kernel With Tree for Channel Reduction\n\n" << endl;
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int mini_batch;

    if(Channel == 1) {
        mini_batch = TILE_WIDTH4;
    } else {
        if(Batch == 100)
            mini_batch = 10;
        else
            mini_batch = 8;
    }
    

    size_t shmem_size = sizeof(float) * (mini_batch * (Channel * (TILE_WIDTH4 + (K - 1)) * (TILE_WIDTH4 + (K - 1)) + Channel * TILE_WIDTH4 * TILE_WIDTH4));

    const int W_size2 = ceil(float(Width_out) / TILE_WIDTH4); 
    const int H_size2 = ceil(float(Height_out) / TILE_WIDTH4);
    const int Y2 = H_size2 * W_size2;
    

    dim3 blockDim2(TILE_WIDTH4, TILE_WIDTH4, Channel * mini_batch);
    dim3 gridDim2(Map_out, Y2, Batch / mini_batch);
    conv_forward_kernel_tiled_tree_reduction<<<gridDim2, blockDim2, shmem_size>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void launchTiledKernelChannelReductionAtomics(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    cout << "\n\nLaunching Tiled Kernel With Atomics for Channel Reduction\n\n" << endl;
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int mini_batch;

    if(Channel == 1) {
        mini_batch = TILE_WIDTH4;
    } else {
        if(Batch == 100)
            mini_batch = 10;
        else
            mini_batch = 8;
    }
    

    size_t shmem_size = sizeof(float) * (mini_batch * (Channel * (TILE_WIDTH4 + (K - 1)) * (TILE_WIDTH4 + (K - 1))));

    const int W_size2 = ceil(float(Width_out) / TILE_WIDTH4); 
    const int H_size2 = ceil(float(Height_out) / TILE_WIDTH4);
    const int Y2 = H_size2 * W_size2;
    

    dim3 blockDim2(TILE_WIDTH4, TILE_WIDTH4, Channel * mini_batch);
    dim3 gridDim2(Map_out, Y2, Batch / mini_batch);
    conv_forward_kernel_tiled_atomics<<<gridDim2, blockDim2, shmem_size>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void launchUnrolledMatrixMultiplicationKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    cout << "\n\nLaunching Unrolled Matrix Multiplication Kernel\n\n" << endl;
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    int H_unroll = Channel * K * K;
    int W_unroll = H_out * W_out;
    float* input_unrolled;
    cudaMalloc((void **) &input_unrolled, W_unroll * H_unroll * sizeof(float));
    // float *mask = (float *)malloc(sizeof(float) * K * K * Map_out);
    // cudaMemcpy(mask, device_mask, sizeof(float) * K * K * Map_out, cudaMemcpyDeviceToHost);
    // for(int i = 0; i < Map_out; i++) {
    //     for(int j =0; j < H_unroll; j++) {
    //         printf("%f ", mask[i * H_unroll + j]);
    //     } printf("|\n");
    // }
    for (int b = 0; b < Batch; b++) {
        dim3 DimBlock_Unroll(1024, 1, 1);
        dim3 DimGrid_Unroll(ceil((Channel * H_out * W_out) / 1024.0), 1, 1);
        
        unroll<<<DimGrid_Unroll, DimBlock_Unroll>>>(b, Channel, Height, Width, K, device_input, input_unrolled);

        // printf("Unrolling done for b = %d\n", b);
        cudaDeviceSynchronize();

        int numCColumns = W_unroll;
        int numCRows = Map_out;

        int blockSizeX = ceil((1.0 * numCColumns) / TILE_WIDTH3);
        int blockSizeY = ceil((1.0 * numCRows) / TILE_WIDTH3);

        dim3 DimBlock_MatMul(TILE_WIDTH3, TILE_WIDTH3, 1);
        dim3 DimGrid_MatMul(blockSizeX, blockSizeY, 1);

        matrixMultiply<<<DimGrid_MatMul, DimBlock_MatMul>>>(input_unrolled, device_output + (b * (W_out * H_out) * Map_out), numCRows, H_unroll, H_unroll,  numCColumns, numCRows, numCColumns);

        // printf("MatMul done for b = %d\n", b);
        
        cudaDeviceSynchronize();
    }
    cudaFree(input_unrolled);
}


__host__ void launchFusedUnrolledMatrixMultiplicationKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    cout << "\n\nLaunching Fused Unrolled Matrix Multiplication Kernel\n\n" << endl;
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    // int H_unroll = Channel * K * K;
    // int W_unroll = H_out * W_out;

    
    dim3 DimBlock_Unroll(64, 1, 4);
    dim3 DimGrid_Unroll(ceil((Channel * H_out * W_out) / 64.0), 1, ceil(Batch / 4.0));
    
    fusedUnrollAndMatMul<<<DimGrid_Unroll, DimBlock_Unroll>>>(Channel, Height, Width, K, device_input, device_output, Map_out);
    
}

__host__ void launchRestrictLoopUnrollKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    cout << "\n\nLaunching Restric and Loop unrolled Kernel with weights in the constant memory \n\n" << endl;
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int W_size = ceil(float(Width_out) / TILE_WIDTH); 
    const int H_size = ceil(float(Height_out) / TILE_WIDTH);
    const int Y = H_size * W_size;
    const int Z = ceil(float(Batch) / TILE_WIDTH);


    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(Map_out, Y, Z);

    restrict_loop_unroll_kernel<<<gridDim, blockDim>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    #ifdef NUM_STREAMS
        return;
    #endif

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Copy the output back to host
    cudaMemcpy( host_output, 
                device_output, 
                Batch * ((Width_out * Height_out) * Map_out) * sizeof(float), 
                cudaMemcpyDeviceToHost);
 
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    // cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
