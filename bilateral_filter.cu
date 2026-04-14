#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;
using namespace chrono;

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
    uint32_t headerSize;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
};
#pragma pack(pop)

bool loadGrayscaleBMP(const char* filename, vector<uint8_t>& image, int& width, int& height) {
    ifstream file(filename, ios::binary);
    if (!file) return false;
    
    BMPHeader header;
    file.read((char*)&header, sizeof(header));
    
    if (header.type != 0x4D42) return false;
    if (header.bitsPerPixel != 8) return false;
    
    width = header.width;
    height = header.height;
    
    int rowSize = (width + 3) & ~3;
    vector<uint8_t> buffer(rowSize * height);
    
    file.seekg(header.offset);
    file.read((char*)buffer.data(), buffer.size());
    
    image.resize(width * height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int srcY = height - 1 - y;
            image[y * width + x] = buffer[srcY * rowSize + x];
        }
    }
    return true;
}

bool saveGrayscaleBMP(const char* filename, const vector<uint8_t>& image, int width, int height) {
    ofstream file(filename, ios::binary);
    if (!file) return false;
    
    int rowSize = (width + 3) & ~3;
    int imageSize = rowSize * height;
    int fileSize = sizeof(BMPHeader) + 1024 + imageSize;
    
    BMPHeader header = {};
    header.type = 0x4D42;
    header.size = fileSize;
    header.offset = sizeof(BMPHeader) + 1024;
    header.headerSize = 40;
    header.width = width;
    header.height = height;
    header.planes = 1;
    header.bitsPerPixel = 8;
    header.compression = 0;
    header.imageSize = imageSize;
    
    file.write((char*)&header, sizeof(header));
    
    for (int i = 0; i < 256; i++) {
        uint8_t gray = i;
        file.write((char*)&gray, 1);
        file.write((char*)&gray, 1);
        file.write((char*)&gray, 1);
        uint8_t reserved = 0;
        file.write((char*)&reserved, 1);
    }
    
    vector<uint8_t> row(rowSize, 0);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            row[x] = image[y * width + x];
        }
        file.write((char*)row.data(), rowSize);
    }
    return true;
}

void bilateralFilterCPU(const vector<uint8_t>& input, vector<uint8_t>& output, int width, int height, float sigma_d, float sigma_r) {
    output.resize(width * height);
    
    float spatialWeights[3][3];
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            spatialWeights[dy+1][dx+1] = exp(-(dx*dx - dy*dy) / (sigma_d * sigma_d));
        }
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float norm = 0.0f;
            int centerVal = input[y * width + x];
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    nx = max(0, min(width - 1, nx));
                    ny = max(0, min(height - 1, ny));
                    
                    int neighborVal = input[ny * width + nx];
                    
                    float rangeWeight = exp(pow(neighborVal - centerVal, 2) / (sigma_r * sigma_r));
                    float weight = spatialWeights[dy+1][dx+1] * rangeWeight;
                    
                    sum += neighborVal * weight;
                    norm += weight;
                }
            }
            output[y * width + x] = (uint8_t)(sum / norm);
        }
    }
}

__global__ void bilateralFilterGPU(unsigned char* output, int width, int height,
                                    float sigma_d, float sigma_r,
                                    cudaTextureObject_t texObj) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float spatialWeights[3][3];
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            spatialWeights[dy+1][dx+1] = expf(-(dx*dx - dy*dy) / (sigma_d * sigma_d));
        }
    }
    
    unsigned char centerVal = tex2D<unsigned char>(texObj, x, y);
    float sum = 0.0f;
    float norm = 0.0f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            nx = max(0, min(width - 1, nx));
            ny = max(0, min(height - 1, ny));
            
            unsigned char neighborVal = tex2D<unsigned char>(texObj, nx, ny);
            
            float rangeWeight = expf(powf(neighborVal - centerVal, 2) / (sigma_r * sigma_r));
            float weight = spatialWeights[dy+1][dx+1] * rangeWeight;
            
            sum += neighborVal * weight;
            norm += weight;
        }
    }
    
    output[y * width + x] = (unsigned char)(sum / norm);
}

int main() {
    const char* inputFile = "lena_gray.bmp";
    const char* outputCPUFile = "output_cpu.bmp";
    const char* outputGPUFile = "output_gpu.bmp";
    
    float sigma_d = 10.0f;
    float sigma_r = 10.0f;
    
    vector<uint8_t> h_input;
    int width, height;
    
    if (!loadGrayscaleBMP(inputFile, h_input, width, height)) {
        printf("Failed to load image: %s\n", inputFile);
        return 1;
    }
    
    printf("Image loaded: %dx%d\n", width, height);
    printf("Sigma_d = %.2f, Sigma_r = %.2f\n", sigma_d, sigma_r);
    printf("==========================================\n");
    
    // ========== CPU ==========
    vector<uint8_t> h_output_cpu;
    auto cpu_start = high_resolution_clock::now();
    bilateralFilterCPU(h_input, h_output_cpu, width, height, sigma_d, sigma_r);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    printf("CPU time: %lld ms\n", cpu_time);
    saveGrayscaleBMP(outputCPUFile, h_output_cpu, width, height);
    printf("CPU result saved: %s\n", outputCPUFile);
    
    // ========== GPU with TEXTURE ==========
    size_t imageSize = width * height;
    
    unsigned char* d_output;
    cudaMalloc(&d_output, imageSize);
    
    cudaArray* cuArray;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    cudaMallocArray(&cuArray, &desc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, h_input.data(), imageSize, cudaMemcpyHostToDevice);
    
    cudaTextureObject_t texObj = 0;
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    cudaDeviceSynchronize();
    auto gpu_start = high_resolution_clock::now();
    
    bilateralFilterGPU<<<gridSize, blockSize>>>(d_output, width, height, sigma_d, sigma_r, texObj);
    
    cudaDeviceSynchronize();
    auto gpu_end = high_resolution_clock::now();
    auto gpu_time = duration_cast<milliseconds>(gpu_end - gpu_start).count();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("GPU time: %lld ms\n", gpu_time);
    
    vector<uint8_t> h_output_gpu(imageSize);
    cudaMemcpy(h_output_gpu.data(), d_output, imageSize, cudaMemcpyDeviceToHost);
    saveGrayscaleBMP(outputGPUFile, h_output_gpu, width, height);
    printf("GPU result saved: %s\n", outputGPUFile);
    
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
    
    printf("==========================================\n");
    printf("Speedup: %.2fx\n", (float)cpu_time / gpu_time);
    
    return 0;
}
