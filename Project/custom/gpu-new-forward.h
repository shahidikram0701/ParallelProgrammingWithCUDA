#ifndef SRC_LAYER_GPU_NEW_FORWARD_H
#define SRC_LAYER_GPU_NEW_FORWARD_H


__host__ void launchBaselineKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);

__host__ void launchTiledKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);

__host__ void launchUnrolledMatrixMultiplicationKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);

__host__ void launchFusedUnrolledMatrixMultiplicationKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);

__host__ void launchTiledKernelChannelTreeReduction(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);

__host__ void launchTiledKernelChannelReductionAtomics(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);

__host__ void conv_forward_gpu_prolog_stream(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);

__host__ void launchRestrictLoopUnrollKernel(float *device_output, const float *device_input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);

using namespace std;

class GPUInterface
{
    public:
    void get_device_properties();
    void conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);
    void conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);
    void conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);
};

#endif
