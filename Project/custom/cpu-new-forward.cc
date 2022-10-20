#include "cpu-new-forward.h"

void conv_forward_cpu(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
  /*
  Modify this function to implement the forward pass described in Chapter 16.
  The code in 16 is for a single image.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct, not fast (this is the CPU implementation.)

  Function paramters:
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

  for(int b = 0; b < Batch; ++b) {
    for(int m = 0; m < Map_out; ++m) {
      for(int output_i = 0; output_i < Height_out; ++output_i) {
        for(int output_j = 0; output_j < Width_out; ++output_j) {
          float conv = 0.0;
          for(int c = 0; c < Channel; ++c) {
            for(int mask_i = 0; mask_i < K; ++mask_i) {
              for(int mask_j = 0; mask_j < K; ++mask_j) {
                conv += (in_4d(b, c, output_i + mask_i, output_j + mask_j) * mask_4d(m, c, mask_i, mask_j));
              }
            }
          }
          out_4d(b, m, output_i, output_j) = conv;
        }
      }
    }
  }

  #undef out_4d
  #undef in_4d
  #undef mask_4d

}

// http://s3.amazonaws.com/files.rai-project.com/userdata/build-6346218db1b11b344de7fb3b.tar.gz