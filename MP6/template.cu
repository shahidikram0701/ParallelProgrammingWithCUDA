// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

__global__
void cast(float* deviceInput, int w, int h, unsigned char* deviceUCharImage) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int c = threadIdx.z;
   int idx = (c * w * h) + (i * w) + j;
   if(j < w and i < h) {
    deviceUCharImage[idx] = (unsigned char) (255 * deviceInput[idx]);
  }
}

__global__
void grayify(unsigned char* deviceUCharImage, int w, int h, int c, unsigned char* deviceUCharGrayImage) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = i * w + j;

  if(j < w and i < h) {
    unsigned char r = deviceUCharImage[c * idx];
		unsigned char g = deviceUCharImage[c * idx + 1];
		unsigned char b = deviceUCharImage[c * idx + 2];
		deviceUCharGrayImage[idx] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__
void histogrammize(unsigned char* deviceUCharGrayImage, int w, int h, unsigned int* deviceBins) {
  __shared__ unsigned int histo[HISTOGRAM_LENGTH];
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = i * w + j;

  int x = (threadIdx.y * blockDim.x + threadIdx.x);

  if(x < HISTOGRAM_LENGTH) {
    histo[x] = 0;
  }
  // printf("idx: %d\n", x);

  __syncthreads();

  if (j < w && i < h) {
      unsigned char val = deviceUCharGrayImage[idx];
      atomicAdd(&(histo[val]), 1);
  }

  __syncthreads();

  if(x < HISTOGRAM_LENGTH) {
   atomicAdd(&deviceBins[x], histo[x]);
  }

}

__global__ 
void cdfy(unsigned int *X, int w, int h, float *Y)  {
  int len = HISTOGRAM_LENGTH;
  __shared__ float XY[HISTOGRAM_LENGTH];
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
    if (index < (HISTOGRAM_LENGTH)) {
      XY[index] += XY[index - stride];
    }
  }
  for (int stride = (HISTOGRAM_LENGTH) / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index + stride < (HISTOGRAM_LENGTH)) {
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads();

  if (i < len) {
    Y[i] = XY[threadIdx.x] / ((float) (w * h));
  }
  if (i + blockDim.x < len) {
    Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x] / ((float) (w * h));
  }
}

__global__
void equalise(unsigned char* input, int w, int h, float* cdf, float cdfmin, float* output) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int c = threadIdx.z;
  int idx = (c * w * h) + (i * w) + j;
   if(j < w and i < h) {
      unsigned char val = input[idx];
      float x = 255 * (cdf[val] - cdfmin) / (1.0 - cdfmin);

      float start = 0.0;
      float end = 255.0;
      output[idx] = (float) ((min(max(x, start), end)) / 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  float* deviceInputImageData;
  unsigned char* deviceUCharImage;
  unsigned char* deviceUCharGrayImage;
  unsigned int* deviceBins;
  float* deviceCDF;
  float* deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_stop(Generic, "Importing data and creating memory on host");

  int n = imageHeight * imageWidth * imageChannels;

  cudaMalloc((void **) &deviceInputImageData, n * sizeof(float));
  cudaMalloc((void **) &deviceUCharImage, n * sizeof(unsigned char));

  cudaMemcpy(deviceInputImageData, hostInputImageData, n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 DimBlockChannels(BLOCK_SIZE, BLOCK_SIZE, imageChannels);
  dim3 DimGridChannels(ceil(float(imageWidth) / BLOCK_SIZE), ceil(float(imageHeight) / BLOCK_SIZE), 1);


  cast<<<DimGridChannels, DimBlockChannels>>>(deviceInputImageData, imageWidth, imageHeight, deviceUCharImage);


  cudaDeviceSynchronize();

  cudaMalloc((void **) &deviceUCharGrayImage, n * sizeof(unsigned char));

  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 DimGrid(ceil(float(imageWidth) / BLOCK_SIZE), ceil(float(imageHeight) / BLOCK_SIZE), 1);

  grayify<<<DimGrid, DimBlock>>>(deviceUCharImage, imageWidth, imageHeight, imageChannels, deviceUCharGrayImage);

  cudaDeviceSynchronize();

  cudaMalloc((void **) &deviceBins, HISTOGRAM_LENGTH * sizeof(unsigned int));

  histogrammize<<<DimGrid, DimBlock>>>(deviceUCharGrayImage, imageWidth, imageHeight, deviceBins);

  cudaDeviceSynchronize();


  cudaMalloc((void **) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));

  dim3 DimGridScan(1, 1, 1);
  dim3 DimBlockScan(HISTOGRAM_LENGTH/2, 1, 1);

  cdfy<<<DimGridScan, DimBlockScan>>>(deviceBins, imageWidth, imageHeight, deviceCDF);

  cudaDeviceSynchronize();

  float hostCDF[HISTOGRAM_LENGTH];
  cudaMemcpy(hostCDF, deviceCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);


  float minCDF = 1.0;

  for(int i = 0; i < HISTOGRAM_LENGTH; ++i) {
    if(hostCDF[i] < minCDF) {
      minCDF = hostCDF[i];
    }
  }

  cudaMalloc((void **) &deviceOutputImageData, n * sizeof(float));

  equalise<<<DimGridChannels, DimBlockChannels>>>(deviceUCharImage, imageWidth, imageHeight, deviceCDF, minCDF, deviceOutputImageData);

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  wbSolution(args, outputImage);


  cudaFree(deviceInputImageData);
  cudaFree(deviceUCharGrayImage);
  cudaFree(deviceBins);
  cudaFree(deviceCDF);
  cudaFree(deviceOutputImageData);


  return 0;
}
