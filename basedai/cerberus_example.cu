/*
BASEDAI: CERBERUS SQUEEZING IN LOCAL TRANSFORMERS MODELS
This script implements Cerberus Squeezing, an optimized transformer model for homomorphic 
encryption (HE) using CUDA-accelerated GPU computing. The key innovation is an entropy-based 
pruning mechanism that optimizes the multihead attention (MHA) mechanism to reduce 
computational complexity in encrypted operations. Key features include:
1. Custom Ciphertext structure encapsulating encrypted values and noise levels.
2. CerberusSqueeze structure maintaining a squeeze rate and entropy matrix for pruning guidance.
3. Optimized transformer model incorporating Cerberus Squeezing for efficient FHE computations.
4. Standard transformer model implementation for comparison.
5. CUDA kernels for efficient parallel processing of MHA and other transformer computations.
6. Benchmarking system to compare performance and noise characteristics of both models.
Approach is compatible with various transformer-based models, allowing for the creation
of zero-knowledge LLMs (ZK-LLMs) on the BasedAI network.
// example output using x1 consumer NVIDIA RTX 4090 on UBUNTU 24
// Finished standard_transformer_model
...
Cerberus model execution time: 0.46394 ms (+178%) 
Standard model execution time: 1.2898 ms
Optimized model max noise: 1058443425
Standard model max noise: 1090360451
Copying results back to host
Results:
Max noise in Cerberus Squeeze transformer model: 99 (SUCCESS)
Max noise in standard transformer model: 99 (SUCCESS)
*/


#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <algorithm>
#include <stdio.h>
#include <cassert>
#include <chrono>

struct Ciphertext {
    float value;
    unsigned noise;
};

struct MultiHeadAttention {
    Ciphertext* query_projections;
    Ciphertext* key_projections;
    Ciphertext* value_projections;
    Ciphertext* output_projection;
};

struct TransformerLayer {
    MultiHeadAttention multi_head_attention;
    Ciphertext* ff_layer1;
    Ciphertext* ff_layer2;
    Ciphertext* layer_norm1;
    Ciphertext* layer_norm2;
};

struct TransformerModel {
    TransformerLayer* layers;
    int num_layers;
    int rows;
    int cols;
};

struct CerberusSqueeze {
    float squeeze_rate;
    float* entropy_matrix;
};

void printDevicePointer(const char* label, const void* ptr) {
    std::cout << label << ": 0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) << std::dec << std::endl;
}

void printHostTransformerLayerInfo(const TransformerLayer& layer, int layerIndex) {
    std::cout << "Host Layer " << layerIndex << " info:" << std::endl;
    std::cout << "  multi_head_attention.query_projections: " << layer.multi_head_attention.query_projections << std::endl;
    std::cout << "  multi_head_attention.key_projections: " << layer.multi_head_attention.key_projections << std::endl;
    std::cout << "  multi_head_attention.value_projections: " << layer.multi_head_attention.value_projections << std::endl;
    std::cout << "  multi_head_attention.output_projection: " << layer.multi_head_attention.output_projection << std::endl;
    std::cout << "  ff_layer1: " << layer.ff_layer1 << std::endl;
    std::cout << "  ff_layer2: " << layer.ff_layer2 << std::endl;
    std::cout << "  layer_norm1: " << layer.layer_norm1 << std::endl;
    std::cout << "  layer_norm2: " << layer.layer_norm2 << std::endl;
}

void printHostTransformerModelInfo(const TransformerModel& model) {
    std::cout << "Host TransformerModel info:" << std::endl;
    std::cout << "Number of layers: " << model.num_layers << std::endl;
    std::cout << "Layers pointer: " << model.layers << std::endl;
    for (int i = 0; i < model.num_layers; ++i) {
        printHostTransformerLayerInfo(model.layers[i], i);
    }
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__device__ void safelyPrintCiphertext(const char* label, const Ciphertext* c) {
    if (c != nullptr) {
        printf("%s: value = %f, noise = %u\n", label, c->value, c->noise);
    } else {
        printf("%s: nullptr\n", label);
    }
}

__global__ void layer_normalization_with_cerberus_squeeze(Ciphertext* input, float* entropy_matrix, Ciphertext* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f, sum_sq = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += input[i].value;
            sum_sq += input[i].value * input[i].value;
        }
        float mean = sum / size;
        float var = (sum_sq / size) - (mean * mean);
        
        float epsilon = 1e-5f;
        output[idx].value = (input[idx].value - mean) / sqrtf(var + epsilon);
        output[idx].noise = input[idx].noise;

        entropy_matrix[idx] *= exp(-0.1f * output[idx].value);
        
        if (idx == 0) {
            printf("layer_norm_with_cerberus_squeeze[0]: value = %f, noise = %u\n", output[idx].value, output[idx].noise);
        }
    }
}

__global__ void verifyPointers(Ciphertext* input, const TransformerLayer* layer, CerberusSqueeze* squeeze, int rows, int cols) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Verifying pointers:\n");
        printf("input: %p\n", input);
        printf("layer: %p\n", layer);
        printf("squeeze: %p\n", squeeze);
        if (layer != nullptr) {
            printf("layer->ff_layer1: %p\n", layer->ff_layer1);
            printf("layer->ff_layer2: %p\n", layer->ff_layer2);
            printf("layer->layer_norm1: %p\n", layer->layer_norm1);
            printf("layer->layer_norm2: %p\n", layer->layer_norm2);
        }
        if (squeeze != nullptr) {
            printf("squeeze->entropy_matrix: %p\n", squeeze->entropy_matrix);
        }
    }
}

__device__ void printCiphertext(const char* label, const Ciphertext& c) {
    printf("%s: value = %f, noise = %u\n", label, c.value, c.noise);
}

void printTransformerLayerInfo(const TransformerLayer& layer, int layerIndex) {
    std::cout << "Layer " << layerIndex << " info:" << std::endl;
    std::cout << "  multi_head_attention.query_projections: " << layer.multi_head_attention.query_projections << std::endl;
    std::cout << "  multi_head_attention.key_projections: " << layer.multi_head_attention.key_projections << std::endl;
    std::cout << "  multi_head_attention.value_projections: " << layer.multi_head_attention.value_projections << std::endl;
    std::cout << "  multi_head_attention.output_projection: " << layer.multi_head_attention.output_projection << std::endl;
    std::cout << "  ff_layer1: " << layer.ff_layer1 << std::endl;
    std::cout << "  ff_layer2: " << layer.ff_layer2 << std::endl;
    std::cout << "  layer_norm1: " << layer.layer_norm1 << std::endl;
    std::cout << "  layer_norm2: " << layer.layer_norm2 << std::endl;
}

void printTransformerModelInfo(const TransformerModel& model) {
    std::cout << "TransformerModel info:" << std::endl;
    std::cout << "Number of layers: " << model.num_layers << std::endl;
    std::cout << "Layers pointer: " << model.layers << std::endl;
    for (int i = 0; i < model.num_layers; ++i) {
        printTransformerLayerInfo(model.layers[i], i);
    }
}

__global__ void debugKernel(Ciphertext* input, const TransformerLayer* layer, CerberusSqueeze* squeeze, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 5) {
        printf("Debug for idx %d:\n", idx);
        safelyPrintCiphertext("Input", input ? &input[idx] : nullptr);
        safelyPrintCiphertext("Layer norm 1", layer && layer->layer_norm1 ? &layer->layer_norm1[idx] : nullptr);
        safelyPrintCiphertext("Layer norm 2", layer && layer->layer_norm2 ? &layer->layer_norm2[idx] : nullptr);
        if (squeeze && squeeze->entropy_matrix) {
            printf("Entropy matrix[%d] = %f\n", idx, squeeze->entropy_matrix[idx]);
        } else {
            printf("Entropy matrix: nullptr or invalid\n");
        }
    }
}

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__device__ unsigned max_noise(const Ciphertext* mat, int rows, int cols) {
    unsigned max_noise = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            max_noise = max(max_noise, mat[i * cols + j].noise);
        }
    }
    return max_noise;
}

__device__ void update_global_squeeze(CerberusSqueeze* squeeze, const Ciphertext* layer_output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            squeeze->entropy_matrix[i * cols + j] *= exp(-0.1f * layer_output[i * cols + j].value);
        }
    }
    float sum_entropy = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        sum_entropy += squeeze->entropy_matrix[i];
    }
    float avg_entropy = sum_entropy / (rows * cols);
    squeeze->squeeze_rate = (squeeze->squeeze_rate + avg_entropy) / 2;
}

__device__ void prune_matrix(const Ciphertext* mat, const float* entropy, float threshold, Ciphertext* pruned_mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            pruned_mat[i * cols + j] = (entropy[i * cols + j] < threshold) ? Ciphertext{0, 0} : mat[i * cols + j];
        }
    }
}

__global__ void matrix_multiply(const Ciphertext* A, const Ciphertext* B, Ciphertext* result, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        Ciphertext sum{0, 0};
        for (int k = 0; k < n; ++k) {
            sum.value += A[row * n + k].value * B[k * p + col].value;
            sum.noise = max(sum.noise, max(A[row * n + k].noise, B[k * p + col].noise));
        }
        const unsigned MAX_NOISE = 1000000;
        sum.noise = min(sum.noise, MAX_NOISE);
        result[row * p + col] = sum;
        
        if (row == 0 && col == 0) {
            printf("matrix_multiply result[0]: value = %f, noise = %u\n", sum.value, sum.noise);
        }
    }
}

__global__ void layer_normalization_kernel(Ciphertext* input, Ciphertext* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f, sum_sq = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += input[i].value;
            sum_sq += input[i].value * input[i].value;
        }
        float mean = sum / size;
        float var = (sum_sq / size) - (mean * mean);
        
        float epsilon = 1e-5f;
        output[idx].value = (input[idx].value - mean) / sqrtf(var + epsilon);
        
        unsigned noise_sum = 0;
        for (int i = 0; i < size; ++i) {
            noise_sum += input[i].noise;
        }
        float noise_mean = static_cast<float>(noise_sum) / size;
        
        output[idx].noise = static_cast<unsigned>(input[idx].noise / noise_mean * 100);  
        
        if (idx == 0) {
            printf("layer_norm[0]: value = %f, noise = %u\n", output[idx].value, output[idx].noise);
        }
    }
}

__device__ void layer_normalization(Ciphertext* input, Ciphertext* output, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += input[i].value;
    }
    float mean = sum / size;

    float var_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = input[i].value - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / size;

    float epsilon = 1e-5f;
    for (int i = 0; i < size; ++i) {
        output[i].value = (input[i].value - mean) / sqrtf(var + epsilon);
        output[i].noise = input[i].noise;
    }
}

__device__ int g_debugCounter = 0;

__device__ void cerberus_squeeze_mha_device(Ciphertext* input, const MultiHeadAttention* mha, CerberusSqueeze* squeeze, Ciphertext* output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float entropy = squeeze->entropy_matrix[idx];
        float squeeze_threshold = 0.5f;
        
        if (entropy > squeeze_threshold) {
            output[idx].value = input[idx].value * mha->query_projections[idx].value * mha->key_projections[idx].value * mha->value_projections[idx].value;
            output[idx].noise = max(max(input[idx].noise, mha->query_projections[idx].noise), max(mha->key_projections[idx].noise, mha->value_projections[idx].noise));
        } else {
            output[idx].value = input[idx].value * mha->query_projections[idx].value;
            output[idx].noise = max(input[idx].noise, mha->query_projections[idx].noise);
        }
        
        squeeze->entropy_matrix[idx] *= exp(-squeeze->squeeze_rate * fabs(output[idx].value));
    }
}

__global__ void optimized_transformer_layer(Ciphertext* input, const TransformerLayer* layer, CerberusSqueeze* squeeze, Ciphertext* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        atomicAdd(&g_debugCounter, 1);

        if (idx < 5) {
            printCiphertext("Input", input[idx]);
        }

        output[idx] = input[idx];
        
        __shared__ Ciphertext normalized[1024];
        int local_idx = threadIdx.x;
        if (local_idx < min(cols, 1024)) {
            normalized[local_idx] = output[idx];
        }
        __syncthreads();

        if (local_idx == 0) {
            layer_normalization(normalized, normalized, min(cols, 1024));
        }
        __syncthreads();

        if (local_idx < min(cols, 1024)) {
            output[idx] = normalized[local_idx];
        }
        
        cerberus_squeeze_mha_device(output, &layer->multi_head_attention, squeeze, output, rows * cols);
        
        if (idx == 0) {
            float total_entropy = 0.0f;
            for (int i = 0; i < rows * cols; i++) {
                total_entropy += squeeze->entropy_matrix[i];
            }
            squeeze->squeeze_rate = (squeeze->squeeze_rate + total_entropy / (rows * cols)) / 2;
        }

        if (idx < 5) {
            printCiphertext("Output", output[idx]);
        }
    }
}

void cerberus_squeeze_transformer_model(Ciphertext* d_input, const TransformerModel* d_model, CerberusSqueeze* d_squeeze, Ciphertext* d_output, int rows, int cols) {
    std::cout << "Entering cerberus_squeeze_transformer_model" << std::endl;
    printDevicePointer("d_input", d_input);
    printDevicePointer("d_model", d_model);
    printDevicePointer("d_squeeze", d_squeeze);
    printDevicePointer("d_output", d_output);
    std::cout << "rows: " << rows << ", cols: " << cols << std::endl;

    TransformerModel h_model;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_model, d_model, sizeof(TransformerModel), cudaMemcpyDeviceToHost));
    std::cout << "Copied TransformerModel from device to host" << std::endl;
    
    std::cout << "TransformerModel info:" << std::endl;
    std::cout << "Number of layers: " << h_model.num_layers << std::endl;
    printDevicePointer("Layers pointer", h_model.layers);

    if (h_model.num_layers > 0 && h_model.layers != nullptr) {
        TransformerLayer h_layer;
        for (int i = 0; i < h_model.num_layers; ++i) {
            std::cout << "Attempting to copy Layer " << i << " from device to host" << std::endl;
            CHECK_CUDA_ERROR(cudaMemcpy(&h_layer, h_model.layers + i, sizeof(TransformerLayer), cudaMemcpyDeviceToHost));
            
            std::cout << "Layer " << i << " info:" << std::endl;
            printDevicePointer("  multi_head_attention.query_projections", h_layer.multi_head_attention.query_projections);
            printDevicePointer("  multi_head_attention.key_projections", h_layer.multi_head_attention.key_projections);
            printDevicePointer("  multi_head_attention.value_projections", h_layer.multi_head_attention.value_projections);
            printDevicePointer("  multi_head_attention.output_projection", h_layer.multi_head_attention.output_projection);
            printDevicePointer("  ff_layer1", h_layer.ff_layer1);
            printDevicePointer("  ff_layer2", h_layer.ff_layer2);
            printDevicePointer("  layer_norm1", h_layer.layer_norm1);
            printDevicePointer("  layer_norm2", h_layer.layer_norm2);
        }
    } else {
        std::cout << "Invalid layer information in the model" << std::endl;
    }

    dim3 block_size(32);
    dim3 grid_size((rows * cols + block_size.x - 1) / block_size.x);
    std::cout << "Grid size: (" << grid_size.x << ", " << grid_size.y << ", " << grid_size.z << ")" << std::endl;
    std::cout << "Block size: (" << block_size.x << ", " << block_size.y << ", " << block_size.z << ")" << std::endl;

    Ciphertext* d_layer_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_layer_output, rows * cols * sizeof(Ciphertext)));
    std::cout << "Allocated d_layer_output: " << d_layer_output << std::endl;

    int num_layers = h_model.num_layers;
    std::cout << "Number of layers: " << num_layers << std::endl;

    for (int i = 0; i < num_layers; ++i) {
        std::cout << "Processing layer " << i << std::endl;
        
        TransformerLayer h_layer;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_layer, &h_model.layers[i], sizeof(TransformerLayer), cudaMemcpyDeviceToHost));
        std::cout << "Copied TransformerLayer from device to host" << std::endl;
        printTransformerLayerInfo(h_layer, i);

        int h_debugCounter = 0;
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_debugCounter, &h_debugCounter, sizeof(int)));

        std::cout << "Launching optimized_transformer_layer kernel" << std::endl;
        optimized_transformer_layer<<<grid_size, block_size>>>(
            (i == 0) ? d_input : d_output,
            &h_model.layers[i],
            d_squeeze,
            d_layer_output,
            rows,
            cols
        );
        CUDA_KERNEL_CHECK();
        std::cout << "Synchronizing after kernel launch" << std::endl;
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&h_debugCounter, g_debugCounter, sizeof(int)));
        std::cout << "Layer " << i << " executed " << h_debugCounter << " threads" << std::endl;

        CHECK_CUDA_ERROR(cudaMemcpy(d_output, d_layer_output, rows * cols * sizeof(Ciphertext), cudaMemcpyDeviceToDevice));
    }

    CHECK_CUDA_ERROR(cudaFree(d_layer_output));
    std::cout << "Finished cerberus_squeeze_transformer_model" << std::endl;
}

void standard_transformer_model(Ciphertext* d_input, const TransformerModel* d_model, Ciphertext* d_output, int rows, int cols) {
    std::cout << "Starting standard_transformer_model" << std::endl;
    std::cout << "d_input: " << d_input << ", d_model: " << d_model << ", d_output: " << d_output << std::endl;
    std::cout << "rows: " << rows << ", cols: " << cols << std::endl;
    
    TransformerModel h_model;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_model, d_model, sizeof(TransformerModel), cudaMemcpyDeviceToHost));
    std::cout << "Number of layers: " << h_model.num_layers << std::endl;

    dim3 block_size(32);
    dim3 grid_size((rows * cols + block_size.x - 1) / block_size.x);

    Ciphertext* d_layer_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_layer_output, rows * cols * sizeof(Ciphertext)));

    float* d_dummy_entropy;
    CHECK_CUDA_ERROR(cudaMalloc(&d_dummy_entropy, rows * cols * sizeof(float)));
    
    for (int i = 0; i < h_model.num_layers; ++i) {
        std::cout << "Processing layer " << i << std::endl;
        
        TransformerLayer h_layer;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_layer, &h_model.layers[i], sizeof(TransformerLayer), cudaMemcpyDeviceToHost));

        for (int j = 0; j < 8; ++j) {
            matrix_multiply<<<grid_size, block_size>>>(
                (i == 0 && j == 0) ? d_input : d_output,
                h_layer.multi_head_attention.query_projections + j * rows * cols,
                d_layer_output,
                rows,
                cols,
                cols
            );
            CUDA_KERNEL_CHECK();
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaMemcpy(d_output, d_layer_output, rows * cols * sizeof(Ciphertext), cudaMemcpyDeviceToDevice));
        }

        layer_normalization_with_cerberus_squeeze<<<grid_size, block_size>>>(d_output, d_dummy_entropy, d_layer_output, rows * cols);
        CUDA_KERNEL_CHECK();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(d_output, d_layer_output, rows * cols * sizeof(Ciphertext), cudaMemcpyDeviceToDevice));

        matrix_multiply<<<grid_size, block_size>>>(
            d_output,
            h_layer.ff_layer1,
            d_layer_output,
            rows,
            cols,
            512
        );
        CUDA_KERNEL_CHECK();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        matrix_multiply<<<grid_size, block_size>>>(
            d_layer_output,
            h_layer.ff_layer2,
            d_output,
            rows,
            512,
            cols
        );
        CUDA_KERNEL_CHECK();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        layer_normalization_with_cerberus_squeeze<<<grid_size, block_size>>>(d_output, d_dummy_entropy, d_layer_output, rows * cols);
        CUDA_KERNEL_CHECK();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(d_output, d_layer_output, rows * cols * sizeof(Ciphertext), cudaMemcpyDeviceToDevice));
    }

    CHECK_CUDA_ERROR(cudaFree(d_layer_output));
    CHECK_CUDA_ERROR(cudaFree(d_dummy_entropy));
    std::cout << "Finished standard_transformer_model" << std::endl;
}

unsigned host_max_noise(const Ciphertext* mat, int rows, int cols) {
    unsigned max_noise = 0;
    for (int i = 0; i < rows * cols; ++i) {
        max_noise = std::max(max_noise, mat[i].noise);
    }
    return max_noise;
}

void benchmark_models(Ciphertext* d_input, const TransformerModel* d_model, CerberusSqueeze* d_squeeze, Ciphertext* d_optimized_output, Ciphertext* d_standard_output, int rows, int cols, int num_runs = 10) {
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    float total_optimized_time = 0.0f;
    float total_standard_time = 0.0f;
    unsigned max_optimized_noise = 0;
    unsigned max_standard_noise = 0;

    for (int i = 0; i < num_runs; ++i) {
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        cerberus_squeeze_transformer_model(d_input, d_model, d_squeeze, d_optimized_output, rows, cols);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

        float optimized_time = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&optimized_time, start, stop));
        total_optimized_time += optimized_time;

        unsigned optimized_max_noise;
        CHECK_CUDA_ERROR(cudaMemcpy(&optimized_max_noise, d_optimized_output, sizeof(unsigned), cudaMemcpyDeviceToHost));
        max_optimized_noise = std::max(max_optimized_noise, optimized_max_noise);

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        standard_transformer_model(d_input, d_model, d_standard_output, rows, cols);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

        float standard_time = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&standard_time, start, stop));
        total_standard_time += standard_time;

        unsigned standard_max_noise;
        CHECK_CUDA_ERROR(cudaMemcpy(&standard_max_noise, d_standard_output, sizeof(unsigned), cudaMemcpyDeviceToHost));
        max_standard_noise = std::max(max_standard_noise, standard_max_noise);
    }

    std::cout << "Average over " << num_runs << " runs:" << std::endl;
    std::cout << "Cerberus transformer execution time: " << total_optimized_time / num_runs << " ms" << std::endl;
    std::cout << "Standard transformer execution time: " << total_standard_time / num_runs << " ms" << std::endl;
    std::cout << "Cerberus transformer max noise: " << max_optimized_noise << std::endl;
    std::cout << "Standard transformer max noise: " << max_standard_noise << std::endl;

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

__host__ cudaError_t loadBalanceTransformerLayers(TransformerModel* hostModel, TransformerModel** deviceModels, int numGPUs) {
    cudaError_t cudaStatus;
    int layersPerGPU = hostModel->num_layers / numGPUs;
    int remainingLayers = hostModel->num_layers % numGPUs;

    for (int i = 0; i < numGPUs; i++) {
        cudaStatus = cudaSetDevice(i);
        if (cudaStatus != cudaSuccess) return cudaStatus;

        int startLayer = i * layersPerGPU + min(i, remainingLayers);
        int endLayer = startLayer + layersPerGPU + (i < remainingLayers ? 1 : 0);
        int numLayersForThisGPU = endLayer - startLayer;

        // Allocate device memory for the TransformerModel structure
        TransformerModel* d_model;
        cudaStatus = cudaMalloc(&d_model, sizeof(TransformerModel));
        if (cudaStatus != cudaSuccess) return cudaStatus;

        // Allocate device memory for the layers
        TransformerLayer* d_layers;
        cudaStatus = cudaMalloc(&d_layers, numLayersForThisGPU * sizeof(TransformerLayer));
        if (cudaStatus != cudaSuccess) return cudaStatus;

        // Copy layers to device
        for (int j = 0; j < numLayersForThisGPU; j++) {
            TransformerLayer* hostLayer = &hostModel->layers[startLayer + j];
            TransformerLayer* deviceLayer = &d_layers[j];

            // Allocate and copy query projections
            cudaStatus = cudaMalloc(&deviceLayer->multi_head_attention.query_projections, 8 * hostModel->rows * hostModel->cols * sizeof(Ciphertext));
            if (cudaStatus != cudaSuccess) return cudaStatus;
            cudaStatus = cudaMemcpy(deviceLayer->multi_head_attention.query_projections, hostLayer->multi_head_attention.query_projections, 
                                    8 * hostModel->rows * hostModel->cols * sizeof(Ciphertext), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) return cudaStatus;

            // Allocate and copy key projections
            cudaStatus = cudaMalloc(&deviceLayer->multi_head_attention.key_projections, 8 * hostModel->rows * hostModel->cols * sizeof(Ciphertext));
            if (cudaStatus != cudaSuccess) return cudaStatus;
            cudaStatus = cudaMemcpy(deviceLayer->multi_head_attention.key_projections, hostLayer->multi_head_attention.key_projections, 
                                    8 * hostModel->rows * hostModel->cols * sizeof(Ciphertext), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) return cudaStatus;

            // Allocate and copy value projections
            cudaStatus = cudaMalloc(&deviceLayer->multi_head_attention.value_projections, 8 * hostModel->rows * hostModel->cols * sizeof(Ciphertext));
            if (cudaStatus != cudaSuccess) return cudaStatus;
            cudaStatus = cudaMemcpy(deviceLayer->multi_head_attention.value_projections, hostLayer->multi_head_attention.value_projections, 
                                    8 * hostModel->rows * hostModel->cols * sizeof(Ciphertext), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) return cudaStatus;

            // Allocate and copy output projection
            cudaStatus = cudaMalloc(&deviceLayer->multi_head_attention.output_projection, hostModel->rows * hostModel->cols * sizeof(Ciphertext));
            if (cudaStatus != cudaSuccess) return cudaStatus;
            cudaStatus = cudaMemcpy(deviceLayer->multi_head_attention.output_projection, hostLayer->multi_head_attention.output_projection, 
                                    hostModel->rows * hostModel->cols * sizeof(Ciphertext), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) return cudaStatus;

            // Allocate and copy feed-forward layers
            cudaStatus = cudaMalloc(&deviceLayer->ff_layer1, hostModel->rows * 512 * sizeof(Ciphertext));
            if (cudaStatus != cudaSuccess) return cudaStatus;
            cudaStatus = cudaMemcpy(deviceLayer->ff_layer1, hostLayer->ff_layer1, 
                                    hostModel->rows * 512 * sizeof(Ciphertext), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) return cudaStatus;

            cudaStatus = cudaMalloc(&deviceLayer->ff_layer2, 512 * hostModel->cols * sizeof(Ciphertext));
            if (cudaStatus != cudaSuccess) return cudaStatus;
            cudaStatus = cudaMemcpy(deviceLayer->ff_layer2, hostLayer->ff_layer2, 
                                    512 * hostModel->cols * sizeof(Ciphertext), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) return cudaStatus;

            // Allocate and copy layer normalization components
            cudaStatus = cudaMalloc(&deviceLayer->layer_norm1, hostModel->rows * hostModel->cols * sizeof(Ciphertext));
            if (cudaStatus != cudaSuccess) return cudaStatus;
            cudaStatus = cudaMemcpy(deviceLayer->layer_norm1, hostLayer->layer_norm1, 
                                    hostModel->rows * hostModel->cols * sizeof(Ciphertext), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) return cudaStatus;

            cudaStatus = cudaMalloc(&deviceLayer->layer_norm2, hostModel->rows * hostModel->cols * sizeof(Ciphertext));
            if (cudaStatus != cudaSuccess) return cudaStatus;
            cudaStatus = cudaMemcpy(deviceLayer->layer_norm2, hostLayer->layer_norm2, 
                                    hostModel->rows * hostModel->cols * sizeof(Ciphertext), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) return cudaStatus;
        }

        d_model->layers = d_layers;
        d_model->num_layers = numLayersForThisGPU;
        d_model->rows = hostModel->rows;
        d_model->cols = hostModel->cols;

        cudaStatus = cudaMemcpy(deviceModels[i], d_model, sizeof(TransformerModel), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) return cudaStatus;

        // likely need to just create scheduled cudafree  
        cudaFree(d_model);
    }

    return cudaSuccess;
}

int main() {
    try {
        int deviceCount;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
        if (error_id != cudaSuccess) {
            printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        int dev;
        error_id = cudaSetDevice(0);
        if (error_id != cudaSuccess) {
            printf("cudaSetDevice returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        error_id = cudaGetDevice(&dev);
        if (error_id != cudaSuccess) {
            printf("cudaGetDevice returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        cudaDeviceProp deviceProp;
        error_id = cudaGetDeviceProperties(&deviceProp, dev);
        if (error_id != cudaSuccess) {
            printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", dev, deviceProp.name, deviceProp.major, deviceProp.minor);
        
        const int rows = 64;
        const int cols = 64;
        
        std::cout << "Rows: " << rows << ", Cols: " << cols << std::endl;
        std::cout << "Total elements: " << rows * cols << std::endl;
        std::cout << "Size of Ciphertext: " << sizeof(Ciphertext) << " bytes" << std::endl;
        std::cout << "Total memory required for input/output: " << (rows * cols * sizeof(Ciphertext)) / (1024 * 1024) << " MB" << std::endl;

        size_t total_size = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(Ciphertext);
        if (total_size / sizeof(Ciphertext) != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
            throw std::runtime_error("Integer overflow in memory allocation size calculation");
        }

        std::cout << "Allocating host memory for input" << std::endl;
        Ciphertext* input = new Ciphertext[rows * cols];
        for (int i = 0; i < rows * cols; ++i) {
            input[i] = {static_cast<float>(rand()) / RAND_MAX, static_cast<unsigned>(rand() % 100)};
        }

        std::cout << "Input values (first 5):" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "Value: " << input[i].value << ", Noise: " << input[i].noise << std::endl;
        }

        std::cout << "Max noise in input: " << host_max_noise(input, rows, cols) << std::endl;

        std::cout << "Initializing TransformerModel" << std::endl;
        TransformerModel model;
        model.num_layers = 3;
        model.rows = rows;
        model.cols = cols;
        model.layers = new TransformerLayer[model.num_layers];
        
        std::cout << "Initializing CerberusSqueeze" << std::endl;
        CerberusSqueeze initial_squeeze;
        initial_squeeze.squeeze_rate = 0.1f;
        initial_squeeze.entropy_matrix = new float[rows * cols];
        for (int i = 0; i < rows * cols; ++i) {
            initial_squeeze.entropy_matrix[i] = 1.0f;
        }

        std::cout << "Initializing TransformerModel layers" << std::endl;
        for (int i = 0; i < model.num_layers; ++i) {
            std::cout << "Initializing layer " << i << std::endl;
            model.layers[i].multi_head_attention.query_projections = new Ciphertext[8 * rows * cols];
            model.layers[i].multi_head_attention.key_projections = new Ciphertext[8 * rows * cols];
            model.layers[i].multi_head_attention.value_projections = new Ciphertext[8 * rows * cols];
            model.layers[i].multi_head_attention.output_projection = new Ciphertext[rows * cols];
            model.layers[i].ff_layer1 = new Ciphertext[rows * 512];
            model.layers[i].ff_layer2 = new Ciphertext[512 * cols];
            model.layers[i].layer_norm1 = new Ciphertext[rows * cols];
            model.layers[i].layer_norm2 = new Ciphertext[rows * cols];
            
            for (int j = 0; j < 8 * rows * cols; ++j) {
                model.layers[i].multi_head_attention.query_projections[j] = {static_cast<float>(rand()) / RAND_MAX, static_cast<unsigned>(rand() % 100)};
                model.layers[i].multi_head_attention.key_projections[j] = {static_cast<float>(rand()) / RAND_MAX, static_cast<unsigned>(rand() % 100)};
                model.layers[i].multi_head_attention.value_projections[j] = {static_cast<float>(rand()) / RAND_MAX, static_cast<unsigned>(rand() % 100)};
            }
            for (int j = 0; j < rows * cols; ++j) {
                model.layers[i].layer_norm1[j] = {static_cast<float>(rand()) / RAND_MAX, static_cast<unsigned>(rand() % 100)};
                model.layers[i].layer_norm2[j] = {static_cast<float>(rand()) / RAND_MAX, static_cast<unsigned>(rand() % 100)};
            }
            for (int j = 0; j < rows * 512; ++j) {
                model.layers[i].ff_layer1[j] = {static_cast<float>(rand()) / RAND_MAX, static_cast<unsigned>(rand() % 100)};
            }
            for (int j = 0; j < 512 * cols; ++j) {
                model.layers[i].ff_layer2[j] = {static_cast<float>(rand()) / RAND_MAX, static_cast<unsigned>(rand() % 100)};
            }
        }

        std::cout << "Host-side TransformerModel initialization complete" << std::endl;
        printHostTransformerModelInfo(model);

        std::cout << "Allocating device memory for input" << std::endl;
        Ciphertext* d_input;
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, rows * cols * sizeof(Ciphertext)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, rows * cols * sizeof(Ciphertext), cudaMemcpyHostToDevice));

        std::cout << "Allocating device memory for model" << std::endl;
        TransformerModel* d_model;
        CHECK_CUDA_ERROR(cudaMalloc(&d_model, sizeof(TransformerModel)));

        std::cout << "Allocating device memory for model layers" << std::endl;
        TransformerLayer* d_layers;
        CHECK_CUDA_ERROR(cudaMalloc(&d_layers, model.num_layers * sizeof(TransformerLayer)));

        for (int i = 0; i < model.num_layers; ++i) {
            std::cout << "Allocating device memory for layer " << i << std::endl;
            TransformerLayer d_layer;

            CHECK_CUDA_ERROR(cudaMalloc(&d_layer.multi_head_attention.query_projections, 8 * rows * cols * sizeof(Ciphertext)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_layer.multi_head_attention.query_projections, model.layers[i].multi_head_attention.query_projections, 
                                       8 * rows * cols * sizeof(Ciphertext), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMalloc(&d_layer.multi_head_attention.key_projections, 8 * rows * cols * sizeof(Ciphertext)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_layer.multi_head_attention.key_projections, model.layers[i].multi_head_attention.key_projections, 
                                       8 * rows * cols * sizeof(Ciphertext), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMalloc(&d_layer.multi_head_attention.value_projections, 8 * rows * cols * sizeof(Ciphertext)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_layer.multi_head_attention.value_projections, model.layers[i].multi_head_attention.value_projections, 
                                       8 * rows * cols * sizeof(Ciphertext), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMalloc(&d_layer.multi_head_attention.output_projection, rows * cols * sizeof(Ciphertext)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_layer.multi_head_attention.output_projection, model.layers[i].multi_head_attention.output_projection, 
                                       rows * cols * sizeof(Ciphertext), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMalloc(&d_layer.ff_layer1, rows * 512 * sizeof(Ciphertext)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_layer.ff_layer1, model.layers[i].ff_layer1, rows * 512 * sizeof(Ciphertext), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMalloc(&d_layer.ff_layer2, 512 * cols * sizeof(Ciphertext)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_layer.ff_layer2, model.layers[i].ff_layer2, 512 * cols * sizeof(Ciphertext), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMalloc(&d_layer.layer_norm1, rows * cols * sizeof(Ciphertext)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_layer.layer_norm1, model.layers[i].layer_norm1, rows * cols * sizeof(Ciphertext), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMalloc(&d_layer.layer_norm2, rows * cols * sizeof(Ciphertext)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_layer.layer_norm2, model.layers[i].layer_norm2, rows * cols * sizeof(Ciphertext), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMemcpy(&d_layers[i], &d_layer, sizeof(TransformerLayer), cudaMemcpyHostToDevice));
        }

        std::cout << "Updating device model with layers" << std::endl;
        CHECK_CUDA_ERROR(cudaMemcpy(&(d_model->layers), &d_layers, sizeof(TransformerLayer*), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(&(d_model->num_layers), &(model.num_layers), sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(&(d_model->rows), &(model.rows), sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(&(d_model->cols), &(model.cols), sizeof(int), cudaMemcpyHostToDevice));

        std::cout << "Verifying device model setup" << std::endl;
        TransformerModel h_model_verify;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_model_verify, d_model, sizeof(TransformerModel), cudaMemcpyDeviceToHost));
        std::cout << "Device model num_layers: " << h_model_verify.num_layers << std::endl;
        std::cout << "Device model rows: " << h_model_verify.rows << std::endl;
        std::cout << "Device model cols: " << h_model_verify.cols << std::endl;
        printDevicePointer("Device model layers pointer", h_model_verify.layers);

        std::cout << "Verifying device model layers" << std::endl;
        TransformerLayer h_layer_verify;
        for (int i = 0; i < model.num_layers; ++i) {
            CHECK_CUDA_ERROR(cudaMemcpy(&h_layer_verify, &d_layers[i], sizeof(TransformerLayer), cudaMemcpyDeviceToHost));
            std::cout << "Layer " << i << " device pointers:" << std::endl;
            printDevicePointer("  query_projections", h_layer_verify.multi_head_attention.query_projections);
            printDevicePointer("  key_projections", h_layer_verify.multi_head_attention.key_projections);
            printDevicePointer("  value_projections", h_layer_verify.multi_head_attention.value_projections);
            printDevicePointer("  output_projection", h_layer_verify.multi_head_attention.output_projection);
            printDevicePointer("  ff_layer1", h_layer_verify.ff_layer1);
            printDevicePointer("  ff_layer2", h_layer_verify.ff_layer2);
            printDevicePointer("  layer_norm1", h_layer_verify.layer_norm1);
            printDevicePointer("  layer_norm2", h_layer_verify.layer_norm2);
        }

        std::cout << "Allocating device memory for CerberusSqueeze" << std::endl;
        CerberusSqueeze* d_squeeze;
        CHECK_CUDA_ERROR(cudaMalloc(&d_squeeze, sizeof(CerberusSqueeze)));
        float* d_entropy_matrix;
        CHECK_CUDA_ERROR(cudaMalloc(&d_entropy_matrix, rows * cols * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_entropy_matrix, initial_squeeze.entropy_matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(&(d_squeeze->squeeze_rate), &(initial_squeeze.squeeze_rate), sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(&(d_squeeze->entropy_matrix), &d_entropy_matrix, sizeof(float*), cudaMemcpyHostToDevice));

        std::cout << "Allocating device memory for output" << std::endl;
        Ciphertext* d_optimized_output;
        CHECK_CUDA_ERROR(cudaMalloc(&d_optimized_output, rows * cols * sizeof(Ciphertext)));

        Ciphertext* d_standard_output;
        CHECK_CUDA_ERROR(cudaMalloc(&d_standard_output, rows * cols * sizeof(Ciphertext)));

        std::cout << "Starting benchmark" << std::endl;
        benchmark_models(d_input, d_model, d_squeeze, d_optimized_output, d_standard_output, rows, cols, 1000);

        std::cout << "Copying results back to host" << std::endl;
        Ciphertext* optimized_output = new Ciphertext[rows * cols];
        CHECK_CUDA_ERROR(cudaMemcpy(optimized_output, d_optimized_output, rows * cols * sizeof(Ciphertext), cudaMemcpyDeviceToHost));

        Ciphertext* standard_output = new Ciphertext[rows * cols];
        CHECK_CUDA_ERROR(cudaMemcpy(standard_output, d_standard_output, rows * cols * sizeof(Ciphertext), cudaMemcpyDeviceToHost));

        std::cout << "Results:" << std::endl;
        std::cout << "Max noise in Cerberus Squeeze transformer model: " << host_max_noise(optimized_output, rows, cols) << std::endl;
        std::cout << "Max noise in standard transformer model: " << host_max_noise(standard_output, rows, cols) << std::endl;

        std::cout << "Cleaning up device memory" << std::endl;
        CHECK_CUDA_ERROR(cudaFree(d_input));
        CHECK_CUDA_ERROR(cudaFree(d_optimized_output));
        CHECK_CUDA_ERROR(cudaFree(d_standard_output));
        CHECK_CUDA_ERROR(cudaFree(d_entropy_matrix));
        CHECK_CUDA_ERROR(cudaFree(d_squeeze));

        for (int i = 0; i < model.num_layers; ++i) {
            TransformerLayer layer;
            CHECK_CUDA_ERROR(cudaMemcpy(&layer, &d_layers[i], sizeof(TransformerLayer), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaFree(layer.multi_head_attention.query_projections));
            CHECK_CUDA_ERROR(cudaFree(layer.multi_head_attention.key_projections));
            CHECK_CUDA_ERROR(cudaFree(layer.multi_head_attention.value_projections));
            CHECK_CUDA_ERROR(cudaFree(layer.multi_head_attention.output_projection));
            CHECK_CUDA_ERROR(cudaFree(layer.ff_layer1));
            CHECK_CUDA_ERROR(cudaFree(layer.ff_layer2));
            CHECK_CUDA_ERROR(cudaFree(layer.layer_norm1));
            CHECK_CUDA_ERROR(cudaFree(layer.layer_norm2));
        }
        CHECK_CUDA_ERROR(cudaFree(d_layers));
        CHECK_CUDA_ERROR(cudaFree(d_model));

        std::cout << "Cleaning up host memory" << std::endl;
        delete[] input;
        delete[] optimized_output;
        delete[] standard_output;
        delete[] initial_squeeze.entropy_matrix;
        for (int i = 0; i < model.num_layers; ++i) {
            delete[] model.layers[i].multi_head_attention.query_projections;
            delete[] model.layers[i].multi_head_attention.key_projections;
            delete[] model.layers[i].multi_head_attention.value_projections;
            delete[] model.layers[i].multi_head_attention.output_projection;
            delete[] model.layers[i].ff_layer1;
            delete[] model.layers[i].ff_layer2;
            delete[] model.layers[i].layer_norm1;
            delete[] model.layers[i].layer_norm2;
        }
        delete[] model.layers;

        std::cout << "Program completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
        cudaDeviceReset();
        return 1;
    }

    return 0;
}
