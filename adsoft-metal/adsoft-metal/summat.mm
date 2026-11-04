//
//  summar.mm
//  adsoft-metal
//
//  Created by Adolfo Centeno on 01/11/26.
//
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>

#define MATRIX_SIZE 512


int main() {

    // Initialize matrices with some values

    float A[MATRIX_SIZE][MATRIX_SIZE];
    float B[MATRIX_SIZE][MATRIX_SIZE];
    float C[MATRIX_SIZE][MATRIX_SIZE];

    for (int i = 0; i < MATRIX_SIZE; i++) {
      for (int j = 0; j < MATRIX_SIZE; j++) {
        A[i][j] = static_cast<float>(i + j);
        B[i][j] = static_cast<float>(i - j);
      }
    }

    
    @autoreleasepool {

        // Initialize Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        
        if (device == nil) {
            std::cerr << "Metal is not supported on this device."
                      << std::endl;
            return -1;
        }

        // Create command queue
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        if (commandQueue == nil) {
            std::cerr << "Failed to create command queue."
                      << std::endl;
            return -1;
        }

        // Display GPU information
        std::cout << "Metal initialized successfully!"
                  << std::endl;

        std::cout << "GPU: "
                  << [[device name] UTF8String]
                  << std::endl;
        
        
        // Create buffers for the matrices
        id<MTLBuffer> bufferA = [device newBufferWithLength:MATRIX_SIZE * MATRIX_SIZE *
        sizeof(float) options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> bufferB = [device newBufferWithLength:MATRIX_SIZE * MATRIX_SIZE *
        sizeof(float) options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> bufferC = [device newBufferWithLength:MATRIX_SIZE * MATRIX_SIZE *
        sizeof(float) options:MTLResourceStorageModeShared];
        
        // Copy data to buffers
        memcpy([bufferA contents], A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        memcpy([bufferB contents], B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        
        
        // Load and compile the shader
        
        NSError *error = nil;

        id<MTLLibrary> library = [device newDefaultLibrary];

        if (library == nil) {
            std::cerr << "Failed to load default Metal library."
                      << std::endl;
            return -1;
        }
        
        

        id<MTLFunction> computeFunction =
            [library newFunctionWithName:@"matrix_add"];

        if (computeFunction == nil) {
            std::cerr << "Could not find matrix_add kernel."
                      << std::endl;
            return -1;
        }

        id<MTLComputePipelineState> pipelineState =
            [device newComputePipelineStateWithFunction:computeFunction
                                                  error:&error];

        if (pipelineState == nil) {
            std::cerr << "Pipeline error: "
                      << [[error localizedDescription] UTF8String]
                      << std::endl;
            return -1;
        }
        
        // Create a command queue and buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        // Create the compute command encoder
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer
        computeCommandEncoder];
        
        [computeEncoder setComputePipelineState:pipelineState];
        [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
        
        // Define thread group sizes
        MTLSize gridSize = MTLSizeMake(MATRIX_SIZE * MATRIX_SIZE, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1); // Launch 256 threads per group
     
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
     
        // End encoding and commit the command
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        memcpy(C, [bufferC contents], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        
        bool valid = true;

        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {

                float expected = A[i][j] + B[i][j];

                if (std::abs(C[i][j] - expected) > 0.0001f) {
                    valid = false;
                    break;
                }
            }

            if (!valid) break;
        }

        std::cout << "Matrix addition: "
                  << (valid ? "SUCCESS" : "FAILED")
                  << std::endl;

        std::cout << "C[0][0] = " << C[0][0] << std::endl;
        std::cout << "C[1][1] = " << C[1][1] << std::endl;
        std::cout << "C[10][10] = " << C[10][10] << std::endl;
        std::cout << "C[20][20] = " << C[20][20] << std::endl;

    }

    return 0;
}
