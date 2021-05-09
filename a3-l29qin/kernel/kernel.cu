#include <stdio.h>

const int INPUT_DIM = 100;
const int FILTER_DIM= 5; // should be factor of INPUT_DIM
const int CONV_OUT_DIM = INPUT_DIM / FILTER_DIM;
const int CONV_LAYER_SIZE = 10;
const int OUT_NEURON_DIM = CONV_OUT_DIM * CONV_OUT_DIM * CONV_LAYER_SIZE;
const int OUT_LAYER_SIZE = 10;

extern "C" __global__ void convolution_layer(double* input, double* conv_filters, double* outputs) {
    for(int i = 0; i < INPUT_DIM; i += FILTER_DIM) {
        for(int j = 0; j < INPUT_DIM; j += FILTER_DIM) {
            double prod = 0;
            for(int x = 0; x < FILTER_DIM; x++) {
                for(int y = 0; y < FILTER_DIM; y++) {
                    prod += input[(i + x) * INPUT_DIM + j + y] * conv_filters[blockIdx.x * FILTER_DIM * FILTER_DIM + x * FILTER_DIM + y];
                }
            }
            outputs[blockIdx.x * CONV_OUT_DIM * CONV_OUT_DIM + i / FILTER_DIM * CONV_OUT_DIM + j / FILTER_DIM] = prod;
        }
    }
}

extern "C" __global__ void relu_layer(double* conv_out) {
    for(int i = 0; i < CONV_OUT_DIM; i++) {
        for(int j = 0; j < CONV_OUT_DIM; j++) {
            if(conv_out[blockIdx.x * CONV_OUT_DIM * CONV_OUT_DIM + i * CONV_OUT_DIM + j] < 0.0) {
                conv_out[blockIdx.x * CONV_OUT_DIM * CONV_OUT_DIM + i * CONV_OUT_DIM + j] = 0.0;
            }
        }
    }
    
}

extern "C" __global__ void output_layer(double* input, double* weights, double* output) {
    double prod = 0;
    for(int x = 0; x < OUT_NEURON_DIM; x++) {
        double weight = weights[blockIdx.x * OUT_NEURON_DIM + x];
        prod += weight * input[x];
    }
    output[blockIdx.x] = prod;
}

// extern "C" __global__ void debug(const double* info) {
//     printf("WTF %.2f \n", info[0]);
// }

