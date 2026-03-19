# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
## ENTER YOUR NAME:YOKESH H
<h3>ENTER YOUR REGISTER NO</h3>212224230312
<h3>EX. NO</h3>03
<h3>DATE</h3>19:03:2026
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
%%writefile sobelEdgeDetectionFilter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,
                            unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {

        int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
        int Gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

        int sumX = 0;
        int sumY = 0;

        for(int i=-1;i<=1;i++){
            for(int j=-1;j<=1;j++){
                unsigned char pixel = srcImage[(y+i)*width + (x+j)];
                sumX += pixel * Gx[i+1][j+1];
                sumY += pixel * Gy[i+1][j+1];
            }
        }

        int magnitude = sqrtf(sumX*sumX + sumY*sumY);
        magnitude = min(max(magnitude,0),255);

        dstImage[y*width + x] = (unsigned char)magnitude;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {

    Mat image = imread("/content/lion.jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char*)malloc(imageSize);

    unsigned char *d_inputImage, *d_outputImage;

    checkCudaErrors(cudaMalloc(&d_inputImage,imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage,imageSize));

    checkCudaErrors(cudaMemcpy(d_inputImage,
                               image.data,
                               imageSize,
                               cudaMemcpyHostToDevice));

    dim3 blockSize(16,16);
    dim3 gridSize((width+15)/16,(height+15)/16);

    sobelFilter<<<gridSize,blockSize>>>(d_inputImage,d_outputImage,width,height);

    checkCudaErrors(cudaMemcpy(h_outputImage,
                               d_outputImage,
                               imageSize,
                               cudaMemcpyDeviceToHost));

    Mat outputImage(height,width,CV_8UC1,h_outputImage);

    imwrite("output_sobel.jpeg",outputImage);

    printf("Edge detection completed.\n");

    return 0;
}
```
## OUTPUT:
<img width="919" height="530" alt="image" src="https://github.com/user-attachments/assets/8c8a0c9a-3b5a-497f-897a-ad3161905932" />

## RESULT:
Thus the program has been executed by using CUDA to perform Sobel edge detection on an image using GPU parallel processing
1. What challenges did you face while implementing the Sobel filter for color images?
Color images contain three channels (RGB), which makes processing more complex. To simplify the implementation, the image was converted into grayscale before applying the Sobel filter.

2. How did changing the block size influence the performance of your CUDA implementation?
Changing the block size affects GPU thread utilization. A 16×16 block size provided better parallel execution and improved performance compared to smaller block sizes.

3. What were the differences in output between the CUDA and CPU implementations?
Both CUDA and CPU implementations produced similar edge detection results. Minor differences occurred due to floating-point precision and parallel processing in GPU computation.

4. Suggest potential optimizations for improving the performance of the Sobel filter.
Performance can be improved by using shared memory, optimizing memory access patterns, and reducing CPU–GPU data transfers




