#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8


__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void **)device_input_ptr, Batch * ((Width * Height) * Channel) * sizeof(float));

    cudaMalloc((void **)device_output_ptr, Batch * ((Width_out * Height_out) * Map_out) * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, Map_out * ((K * K) * Channel) * sizeof(float));

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

    cudaMemcpy(*device_mask_ptr, 
                host_mask, 
                Map_out * ((K * K) * Channel) * sizeof(float), 
                cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int W_size = ceil(float(Width_out) / TILE_WIDTH); 
    const int H_size = ceil(float(Height_out) / TILE_WIDTH);
    const int Y = H_size * W_size;
    const int Z = ceil(float(Batch) / TILE_WIDTH);


    printf("X: %d\nY: %d\nZ: %d\n\n", Map_out, Y, Z);
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(Map_out, Y, Z);

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);


}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
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
    cudaFree(device_mask);

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
