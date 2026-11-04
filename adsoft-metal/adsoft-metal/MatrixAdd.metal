//
//  MatrixAdd.metal
//  adsoft-metal
//
//  Created by Adolfo Centeno on 01/11/26.
//

#include <metal_stdlib>
using namespace metal;

#define MATRIX_SIZE 512

kernel void matrix_add(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < MATRIX_SIZE * MATRIX_SIZE) {
        C[id] = A[id] + B[id];
    }
}
