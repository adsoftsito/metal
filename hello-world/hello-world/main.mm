//
//  main.cpp
//  hello-world
//
//  Created by Adolfo Centeno on 04/11/26.
//

#include <iostream>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

struct DenseParams {
    uint32_t rows;
    uint32_t inputColumns;
    uint32_t outputColumns;
};

struct ActivationParams {
    uint32_t count;
    uint32_t activation;
};

static id<MTLBuffer> makeBuffer(id<MTLDevice> device,
                                const void *data,
                                NSUInteger length) {
    id<MTLBuffer> buffer = [device newBufferWithLength:length
                                               options:MTLResourceStorageModeShared];
    if (data != nullptr && length > 0) {
        memcpy([buffer contents], data, length);
    }
    return buffer;
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                            id<MTLLibrary> library,
                                            NSString *name,
                                            NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        return nil;
    }
    return [device newComputePipelineStateWithFunction:function error:error];
}

static void dispatch(id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> state,
                     const std::vector<std::pair<id<MTLBuffer>, NSUInteger>> &buffers,
                     std::vector<std::pair<const void *, NSUInteger>> parameters,
                     NSUInteger count) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    for (NSUInteger index = 0; index < buffers.size(); ++index) {
        [encoder setBuffer:buffers[index].first
                    offset:buffers[index].second
                   atIndex:index];
    }
    for (NSUInteger index = 0; index < parameters.size(); ++index) {
        [encoder setBytes:parameters[index].first
                   length:parameters[index].second
                  atIndex:buffers.size() + index];
    }
    MTLSize grid = MTLSizeMake(count, 1, 1);
    NSUInteger width = state.maxTotalThreadsPerThreadgroup;
    if (width > 256) {
        width = 256;
    }
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::cerr << "Metal is not supported on this device.\n";
            return -1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newDefaultLibrary];
        if (library == nil) {
            std::cerr << "Failed to load the Metal neural network library.\n";
            return -1;
        }

        const uint32_t sampleCount = 20;
        const float inputData[40] = {
            -2.848f, 5.975f, -10.429f, -8.024f, -1.284f, 2.872f,
            4.758f, 5.955f, -3.938f, -5.78f, 8.525f, 17.866f,
            -13.715f, 4.789f, -3.19f, 13.309f, 3.227f, 18.344f,
            -5.014f, -17.507f, 0.491f, 1.897f, -0.54f, 1.12f,
            -2.032f, 1.092f, -1.476f, -1.606f, -0.084f, 3.386f,
            -1.618f, 1.16f, 2.238f, -1.671f, -2.102f, 1.924f,
            -1.059f, 4.297f, 2.642f, 2.981f
        };
        float targetData[20];
        for (uint32_t i = 0; i < sampleCount; ++i) {
            targetData[i] = i < 10 ? 0.0f : 1.0f;
        }

        const uint32_t widths[] = {2, 4, 8, 1};
        id<MTLBuffer> weights[3];
        id<MTLBuffer> biases[3];
        id<MTLBuffer> activations[4];
        id<MTLBuffer> deltas[3];
        id<MTLBuffer> weightGradients[3];
        id<MTLBuffer> biasGradients[3];

        for (uint32_t layer = 0; layer < 3; ++layer) {
            NSUInteger weightCount = widths[layer] * widths[layer + 1];
            weights[layer] = makeBuffer(device, nil,
                                        weightCount * sizeof(float));
            biases[layer] = makeBuffer(device, nil,
                                       widths[layer + 1] * sizeof(float));
            deltas[layer] = makeBuffer(device, nil,
                                       sampleCount * widths[layer + 1] * sizeof(float));
            weightGradients[layer] = makeBuffer(device, nil,
                                                weightCount * sizeof(float));
            biasGradients[layer] = makeBuffer(device, nil,
                                              widths[layer + 1] * sizeof(float));
            float *weightData = (float *)[weights[layer] contents];
            float *biasData = (float *)[biases[layer] contents];
            for (NSUInteger i = 0; i < weightCount; ++i) {
                weightData[i] = 0.75f;
            }
            for (NSUInteger i = 0; i < widths[layer + 1]; ++i) {
                biasData[i] = 0.25f;
            }
        }

        activations[0] = makeBuffer(device, inputData,
                                    sampleCount * widths[0] * sizeof(float));
        for (uint32_t i = 1; i < 4; ++i) {
            activations[i] = makeBuffer(device, nil,
                                        sampleCount * widths[i] * sizeof(float));
        }
        id<MTLBuffer> targets = makeBuffer(device, targetData,
                                           sampleCount * sizeof(float));

        id<MTLComputePipelineState> dense = pipeline(device, library,
                                                     @"dense_forward", &error);
        id<MTLComputePipelineState> activate = pipeline(device, library,
                                                        @"activate", &error);
        id<MTLComputePipelineState> outputDelta = pipeline(device, library,
                                                           @"output_delta", &error);
        id<MTLComputePipelineState> hiddenDelta = pipeline(device, library,
                                                           @"hidden_delta", &error);
        id<MTLComputePipelineState> weightGradient = pipeline(device, library,
                                                              @"weight_gradient", &error);
        id<MTLComputePipelineState> biasGradient = pipeline(device, library,
                                                            @"bias_gradient", &error);
        id<MTLComputePipelineState> update = pipeline(device, library,
                                                      @"update_parameters", &error);
        if (dense == nil || activate == nil || outputDelta == nil ||
            hiddenDelta == nil || weightGradient == nil ||
            biasGradient == nil || update == nil) {
            std::cerr << "Failed to compile one or more neural network kernels.\n";
            if (error != nil) {
                std::cerr << [[error localizedDescription] UTF8String] << "\n";
            }
            return -1;
        }

        for (uint32_t layer = 0; layer < 3; ++layer) {
            DenseParams params = {sampleCount, widths[layer], widths[layer + 1]};
            dispatch(queue, dense, {
                {activations[layer], 0},
                {weights[layer], 0},
                {biases[layer], 0},
                {activations[layer + 1], 0}
            }, {{&params, sizeof(params)}},
                     sampleCount * widths[layer + 1]);

            ActivationParams activationParams = {
                sampleCount * widths[layer + 1],
                layer == 2 ? 1u : 0u
            };
            dispatch(queue, activate, {{activations[layer + 1], 0}},
                     {{&activationParams, sizeof(activationParams)}},
                     activationParams.count);
        }

        dispatch(queue, outputDelta, {
            {activations[3], 0},
            {targets, 0},
            {deltas[2], 0}
        }, {{&sampleCount, sizeof(sampleCount)}}, sampleCount);

        for (int32_t layer = 2; layer >= 0; --layer) {
            if (layer < 2) {
                DenseParams params = {
                    sampleCount,
                    widths[layer + 2],
                    widths[layer + 1]
                };
                dispatch(queue, hiddenDelta, {
                    {deltas[layer + 1], 0},
                    {weights[layer + 1], 0},
                    {activations[layer + 1], 0},
                    {deltas[layer], 0}
                }, {{&params, sizeof(params)}}, sampleCount * widths[layer + 1]);
            }

            DenseParams gradientParams = {
                sampleCount, widths[layer], widths[layer + 1]
            };
            dispatch(queue, weightGradient, {
                {activations[layer], 0},
                {deltas[layer], 0},
                {weightGradients[layer], 0}
            }, {{&gradientParams, sizeof(gradientParams)}},
                     widths[layer] * widths[layer + 1]);
            dispatch(queue, biasGradient, {
                {deltas[layer], 0},
                {biasGradients[layer], 0}
            }, {{&gradientParams, sizeof(gradientParams)}},
                     widths[layer + 1]);

            const float learningRate = 0.001f;
            dispatch(queue, update, {
                {weights[layer], 0},
                {weightGradients[layer], 0},
                {biases[layer], 0},
                {biasGradients[layer], 0}
            }, {{&gradientParams, sizeof(gradientParams)},
                {&learningRate, sizeof(learningRate)}},
                     widths[layer] * widths[layer + 1]);
        }

        std::cout << "Metal neural network initialized on GPU: "
                  << [[device name] UTF8String] << "\n";
        std::cout << "Architecture: 2-4-8-1, 20 samples, 1 epoch\n";
        std::cout << "Predictions:\n";
        float *predictions = (float *)[activations[3] contents];
        for (uint32_t i = 0; i < sampleCount; ++i) {
            std::cout << predictions[i] << (i + 1 == sampleCount ? '\n' : ' ');
        }
    }
    return 0;
}
