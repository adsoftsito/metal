//
//  main.cpp
//  hello-world
//
//  Created by Adolfo Centeno on 04/11/26.
//

#include <iostream>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main() {

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
    }

    return 0;
}

