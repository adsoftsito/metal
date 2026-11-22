#include <metal_stdlib>
using namespace metal;

struct DenseParams {
    uint rows;
    uint input_columns;
    uint output_columns;
};

struct ActivationParams {
    uint count;
    uint activation;
};

kernel void dense_forward(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant DenseParams &params [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint rows = params.rows;
    uint in_columns = params.input_columns;
    uint out_columns = params.output_columns;
    if (id >= rows * out_columns) {
        return;
    }

    uint row = id / out_columns;
    uint column = id % out_columns;
    float value = bias[column];
    for (uint in = 0; in < in_columns; ++in) {
        value += input[row * in_columns + in] * weights[in * out_columns + column];
    }
    output[id] = value;
}

kernel void activate(
    device float *values [[buffer(0)]],
    constant ActivationParams &params [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.count) {
        return;
    }

    float value = values[id];
    if (params.activation == 0) {
        values[id] = value > 0.0f ? value : 0.0f;
    } else {
        values[id] = 1.0f / (1.0f + exp(-value));
    }
}

kernel void output_delta(
    device const float *prediction [[buffer(0)]],
    device const float *target [[buffer(1)]],
    device float *delta [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) {
        return;
    }
    float prediction_value = prediction[id];
    float mse_derivative = prediction_value - target[id];
    float sigmoid_derivative = prediction_value * (1.0f - prediction_value);
    delta[id] = mse_derivative * sigmoid_derivative;
}

kernel void hidden_delta(
    device const float *next_delta [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *activation [[buffer(2)]],
    device float *delta [[buffer(3)]],
    constant DenseParams &params [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint rows = params.rows;
    uint current_columns = params.output_columns;
    uint next_columns = params.input_columns;
    if (id >= rows * current_columns) {
        return;
    }

    uint row = id / current_columns;
    uint column = id % current_columns;
    float value = 0.0f;
    for (uint next = 0; next < next_columns; ++next) {
        value += next_delta[row * next_columns + next] *
                 weights[column * next_columns + next];
    }
    float activation_derivative = activation[id] > 0.0f ? 1.0f : 0.0f;
    delta[id] = value * activation_derivative;
}

kernel void weight_gradient(
    device const float *input [[buffer(0)]],
    device const float *delta [[buffer(1)]],
    device float *gradient [[buffer(2)]],
    constant DenseParams &params [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint rows = params.rows;
    uint in_columns = params.input_columns;
    uint out_columns = params.output_columns;
    if (id >= in_columns * out_columns) {
        return;
    }

    uint in = id / out_columns;
    uint out = id % out_columns;
    float value = 0.0f;
    for (uint row = 0; row < rows; ++row) {
        value += input[row * in_columns + in] * delta[row * out_columns + out];
    }
    gradient[id] = value;
}

kernel void bias_gradient(
    device const float *delta [[buffer(0)]],
    device float *gradient [[buffer(1)]],
    constant DenseParams &params [[buffer(2)]],
    uint column [[thread_position_in_grid]]
) {
    if (column >= params.output_columns) {
        return;
    }

    float value = 0.0f;
    for (uint row = 0; row < params.rows; ++row) {
        value += delta[row * params.output_columns + column];
    }
    gradient[column] = value / float(params.rows);
}

kernel void update_parameters(
    device float *weights [[buffer(0)]],
    device const float *weight_gradient [[buffer(1)]],
    device float *bias [[buffer(2)]],
    device const float *bias_gradient [[buffer(3)]],
    constant DenseParams &params [[buffer(4)]],
    constant float &learning_rate [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < params.input_columns * params.output_columns) {
        weights[id] -= learning_rate * weight_gradient[id];
    }
    if (id < params.output_columns) {
        bias[id] -= learning_rate * bias_gradient[id];
    }
}
