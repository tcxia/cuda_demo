#include<iostream>
#include<string>
#include<cassert>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>


#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )

cv::Mat imageRGBA;
cv::Mat imageGray;


uchar4 *d_rgbaImage__;
unsigned char *d_grayImage__;

size_t numRows(){
    return imageRGBA.rows;
}

size_t numCols(){
    return imageRGBA.cols;
}

template<typename T>
void check(T err,const char* const func, const char* const file, const int line){
    if (err!=cudaSuccess){
        std::cerr << "CUDA error at: "<< file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

void preProcess(uchar4 **inputImage, unsigned char **grayImage, uchar4 **d_rgbaImage, unsigned char **d_grayImage, const std::string &filename){
    
    checkCudaErrors(cudaFree(0));

    cv::Mat image;
    image = cv::imread(filename.c_str(),CV_LOAD_IMAGE_COLOR);
    if (image.empty()){
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }

    cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

    imageGray.create(image.rows,image.cols,CV_8UC1);

    if (!imageRGBA.isContinuous() || !imageGray.isContinuous()){
        std::cerr << "Images aren't continuous!" << std::endl;
        exit(1);
    }

    *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
    *grayImage = imageGray.ptr<unsigned char>(0);

    const size_t numPixels = numRows() * numCols();
    checkCudaErrors(cudaMalloc(d_rgbaImage,sizeof(uchar4) *numPixels));
    checkCudaErrors(cudaMalloc(d_grayImage,sizeof(unsigned char) *numPixels));

    checkCudaErrors(cudaMemset(*d_grayImage,0,numPixels * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(*d_rgbaImage,*inputImage,sizeof(uchar4) *numPixels,
                              cudaMemcpyHostToDevice));

    d_rgbaImage__ = *d_rgbaImage;
    d_grayImage__ = *d_grayImage;
}

__global__ void rgba_to_grayscale(const uchar4* const rgbaImage, unsigned char* const grayImage, int numRows, int numCols){
    int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
    threadIdx.x;

    if (threadId < numRows * numCols){
        const unsigned char R = rgbaImage[threadId].x;
        const unsigned char G = rgbaImage[threadId].y;
        const unsigned char B = rgbaImage[threadId].z;
        grayImage[threadId] = .299f * R + .587f * G + .114f * B;
    }
}

void postProcess(const std::string& output_file, unsigned char* data_ptr){
    cv::Mat output(numRows(),numCols(),CV_8UC1,(void*) data_ptr);
    cv::imwrite(output_file.c_str(),output);
}

void cleanup(){
    cudaFree(d_rgbaImage__);
    cudaFree(d_grayImage__);
}

int main(int argc, char* argv[]){
    std::string input_file = argv[1];
    std::string output_file = argv[2];

    uchar4 *h_rgbaImage, *d_rgbaImage;
    unsigned char *h_grayImage, *d_grayImage;

    preProcess(&h_rgbaImage,&h_grayImage,&d_rgbaImage,&d_grayImage,input_file);

    int thread = 16;
    int grid = (numRows()*numCols() + thread - 1) / (thread * thread);
    const dim3 blockSize(thread,thread);
    const dim3 gridSize(grid);
    rgba_to_grayscale<<<gridSize,blockSize>>>(d_rgbaImage,d_grayImage,numRows(),numCols());

    cudaDeviceSynchronize();
    //cudaGetLastError();

    size_t numPixels = numRows() * numCols();
    checkCudaErrors(cudaMemcpy(h_grayImage,d_grayImage,sizeof(unsigned char) *numPixels,cudaMemcpyDeviceToHost));

    postProcess(output_file,h_grayImage);

    cleanup();
}
