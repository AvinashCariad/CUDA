#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>



// CUDA code for the feed forward neural net 

/*









 */




// ============================================================
// CUDA error checking
// ============================================================

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err)                   \
                  << " at " << __FILE__                       \
                  << ":" << __LINE__ << std::endl;            \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)


// ============================================================
// Network dimensions
//
// 2 inputs
// 4 hidden neurons
// 1 output
// 4 training examples
// ============================================================

constexpr int BATCH  = 4;
constexpr int INPUT  = 2;
constexpr int HIDDEN = 4;
constexpr int OUTPUT = 1;


// ============================================================
// Device activation functions
// ============================================================

__device__
float sigmoid_device(float x)
{
    return 1.0f / (1.0f + expf(-x));
}


__device__
float relu_device(float x)
{
    return x > 0.0f ? x : 0.0f;
}


__device__
float relu_derivative_device(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}


// ============================================================
// Forward layer 1
//
// Z1 = X W1 + b1
// A1 = ReLU(Z1)
//
// X   : [BATCH, INPUT]
// W1  : [INPUT, HIDDEN]
// b1  : [HIDDEN]
//
// Z1  : [BATCH, HIDDEN]
// A1  : [BATCH, HIDDEN]
// ============================================================

__global__
void forward_layer1(
    const float* X,
    const float* W1,
    const float* b1,
    float* Z1,
    float* A1)
{
    int sample = blockIdx.x * blockDim.x
               + threadIdx.x;

    int neuron = blockIdx.y * blockDim.y
               + threadIdx.y;

    if (sample < BATCH && neuron < HIDDEN)
    {
        float sum = 0.0f;

        for (int i = 0; i < INPUT; ++i)
        {
            sum += X[sample * INPUT + i]
                 * W1[i * HIDDEN + neuron];
        }

        sum += b1[neuron];

        Z1[sample * HIDDEN + neuron] = sum;

        A1[sample * HIDDEN + neuron]
            = relu_device(sum);
    }
}


// ============================================================
// Forward output layer
//
// Z2 = A1 W2 + b2
// Yhat = sigmoid(Z2)
//
// A1    : [BATCH, HIDDEN]
// W2    : [HIDDEN, OUTPUT]
// b2    : [OUTPUT]
//
// Z2    : [BATCH, OUTPUT]
// Yhat  : [BATCH, OUTPUT]
// ============================================================

__global__
void forward_layer2(
    const float* A1,
    const float* W2,
    const float* b2,
    float* Z2,
    float* Yhat)
{
    int sample = blockIdx.x * blockDim.x
               + threadIdx.x;

    if (sample < BATCH)
    {
        float sum = 0.0f;

        for (int h = 0; h < HIDDEN; ++h)
        {
            sum += A1[sample * HIDDEN + h]
                 * W2[h];
        }

        sum += b2[0];

        Z2[sample] = sum;

        Yhat[sample] = sigmoid_device(sum);
    }
}


// ============================================================
// Binary cross entropy
// ============================================================

__global__
void compute_loss(
    const float* Y,
    const float* Yhat,
    float* loss)
{
    int i = blockIdx.x * blockDim.x
          + threadIdx.x;

    if (i < BATCH)
    {
        float p = Yhat[i];

        p = fminf(
                fmaxf(p, 1e-7f),
                1.0f - 1e-7f
            );

        float l =
            -(Y[i] * logf(p)
            + (1.0f - Y[i])
            * logf(1.0f - p));

        atomicAdd(loss, l / BATCH);
    }
}


// ============================================================
// Output gradient
//
// BCE + sigmoid:
//
// dZ2 = (Yhat - Y) / BATCH
// ============================================================

__global__
void output_gradient(
    const float* Yhat,
    const float* Y,
    float* dZ2)
{
    int i = blockIdx.x * blockDim.x
          + threadIdx.x;

    if (i < BATCH)
    {
        dZ2[i] =
            (Yhat[i] - Y[i])
            / static_cast<float>(BATCH);
    }
}


// ============================================================
// dW2
//
// dW2 = A1^T dZ2
//
// A1   : [BATCH, HIDDEN]
// dZ2  : [BATCH]
// dW2  : [HIDDEN]
// ============================================================

__global__
void compute_dW2(
    const float* A1,
    const float* dZ2,
    float* dW2)
{
    int h = blockIdx.x * blockDim.x
          + threadIdx.x;

    if (h < HIDDEN)
    {
        float sum = 0.0f;

        for (int b = 0; b < BATCH; ++b)
        {
            sum +=
                A1[b * HIDDEN + h]
                * dZ2[b];
        }

        dW2[h] = sum;
    }
}


// ============================================================
// db2
// ============================================================

__global__
void compute_db2(
    const float* dZ2,
    float* db2)
{
    if (threadIdx.x == 0 &&
        blockIdx.x == 0)
    {
        float sum = 0.0f;

        for (int b = 0; b < BATCH; ++b)
            sum += dZ2[b];

        db2[0] = sum;
    }
}


// ============================================================
// Hidden gradient
//
// dA1 = dZ2 W2^T
//
// dZ1 = dA1 * ReLU'(Z1)
// ============================================================

__global__
void hidden_gradient(
    const float* dZ2,
    const float* W2,
    const float* Z1,
    float* dZ1)
{
    int sample = blockIdx.x * blockDim.x
               + threadIdx.x;

    int h = blockIdx.y * blockDim.y
          + threadIdx.y;

    if (sample < BATCH && h < HIDDEN)
    {
        float dA1 =
            dZ2[sample] * W2[h];

        dZ1[sample * HIDDEN + h] =
            dA1 *
            relu_derivative_device(
                Z1[sample * HIDDEN + h]
            );
    }
}


// ============================================================
// dW1
//
// dW1 = X^T dZ1
//
// X     : [BATCH, INPUT]
// dZ1   : [BATCH, HIDDEN]
// dW1   : [INPUT, HIDDEN]
// ============================================================

__global__
void compute_dW1(
    const float* X,
    const float* dZ1,
    float* dW1)
{
    int input = blockIdx.x * blockDim.x
              + threadIdx.x;

    int h = blockIdx.y * blockDim.y
          + threadIdx.y;

    if (input < INPUT && h < HIDDEN)
    {
        float sum = 0.0f;

        for (int b = 0; b < BATCH; ++b)
        {
            sum +=
                X[b * INPUT + input]
                * dZ1[b * HIDDEN + h];
        }

        dW1[input * HIDDEN + h] = sum;
    }
}


// ============================================================
// db1
//
// db1 = sum(dZ1, axis=0)
// ============================================================

__global__
void compute_db1(
    const float* dZ1,
    float* db1)
{
    int h = blockIdx.x * blockDim.x
          + threadIdx.x;

    if (h < HIDDEN)
    {
        float sum = 0.0f;

        for (int b = 0; b < BATCH; ++b)
        {
            sum +=
                dZ1[b * HIDDEN + h];
        }

        db1[h] = sum;
    }
}


// ============================================================
// SGD update
// ============================================================

__global__
void update_parameters(
    float* param,
    const float* grad,
    int N,
    float learning_rate)
{
    int i = blockIdx.x * blockDim.x
          + threadIdx.x;

    if (i < N)
    {
        param[i] -= learning_rate * grad[i];
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    // --------------------------------------------------------
    // XOR dataset
    // --------------------------------------------------------

    float h_X[BATCH * INPUT] =
    {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };

    float h_Y[BATCH] =
    {
        0,
        1,
        1,
        0
    };


    // --------------------------------------------------------
    // Initialize weights
    // --------------------------------------------------------

    float h_W1[INPUT * HIDDEN];
    float h_b1[HIDDEN] = {0};

    float h_W2[HIDDEN];
    float h_b2[1] = {0};

    std::mt19937 generator(42);

    std::normal_distribution<float>
        distribution(0.0f, 0.5f);

    for (int i = 0;
         i < INPUT * HIDDEN;
         ++i)
    {
        h_W1[i] = distribution(generator);
    }

    for (int i = 0;
         i < HIDDEN;
         ++i)
    {
        h_W2[i] = distribution(generator);
    }


    // --------------------------------------------------------
    // Device pointers
    // --------------------------------------------------------

    float *d_X;
    float *d_Y;

    float *d_W1;
    float *d_b1;

    float *d_W2;
    float *d_b2;

    float *d_Z1;
    float *d_A1;

    float *d_Z2;
    float *d_Yhat;

    float *d_dZ2;
    float *d_dW2;
    float *d_db2;

    float *d_dZ1;
    float *d_dW1;
    float *d_db1;

    float *d_loss;


    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------

    CUDA_CHECK(cudaMalloc(
        &d_X,
        BATCH * INPUT * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_Y,
        BATCH * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_W1,
        INPUT * HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_b1,
        HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_W2,
        HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_b2,
        sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_Z1,
        BATCH * HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_A1,
        BATCH * HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_Z2,
        BATCH * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_Yhat,
        BATCH * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_dZ2,
        BATCH * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_dW2,
        HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_db2,
        sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_dZ1,
        BATCH * HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_dW1,
        INPUT * HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_db1,
        HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_loss,
        sizeof(float)));


    // --------------------------------------------------------
    // Copy data and initial parameters to GPU
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(
        d_X,
        h_X,
        sizeof(h_X),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_Y,
        h_Y,
        sizeof(h_Y),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_W1,
        h_W1,
        sizeof(h_W1),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_b1,
        h_b1,
        sizeof(h_b1),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_W2,
        h_W2,
        sizeof(h_W2),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_b2,
        h_b2,
        sizeof(h_b2),
        cudaMemcpyHostToDevice));


    // --------------------------------------------------------
    // Kernel configurations
    // --------------------------------------------------------

    dim3 threads2D(16, 16);

    dim3 blocksLayer1(
        (BATCH + threads2D.x - 1)
            / threads2D.x,

        (HIDDEN + threads2D.y - 1)
            / threads2D.y
    );

    dim3 blocksW1(
        (INPUT + threads2D.x - 1)
            / threads2D.x,

        (HIDDEN + threads2D.y - 1)
            / threads2D.y
    );

    const int threads = 256;

    const int batchBlocks =
        (BATCH + threads - 1)
        / threads;

    const int hiddenBlocks =
        (HIDDEN + threads - 1)
        / threads;

    const int w1Elements =
        INPUT * HIDDEN;


    // --------------------------------------------------------
    // Training
    // --------------------------------------------------------

    constexpr int EPOCHS = 20000;

    constexpr float LEARNING_RATE = 0.1f;


    for (int epoch = 0;
         epoch < EPOCHS;
         ++epoch)
    {
        // ====================================================
        // Forward
        // ====================================================

        forward_layer1<<<
            blocksLayer1,
            threads2D
        >>>(
            d_X,
            d_W1,
            d_b1,
            d_Z1,
            d_A1
        );


        forward_layer2<<<
            batchBlocks,
            threads
        >>>(
            d_A1,
            d_W2,
            d_b2,
            d_Z2,
            d_Yhat
        );


        // ====================================================
        // Loss
        // ====================================================

        if (epoch % 1000 == 0)
        {
            CUDA_CHECK(
                cudaMemset(
                    d_loss,
                    0,
                    sizeof(float)
                )
            );

            compute_loss<<<
                batchBlocks,
                threads
            >>>(
                d_Y,
                d_Yhat,
                d_loss
            );

            CUDA_CHECK(
                cudaGetLastError()
            );

            float h_loss;

            CUDA_CHECK(
                cudaMemcpy(
                    &h_loss,
                    d_loss,
                    sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );

            std::cout
                << "Epoch "
                << epoch
                << " Loss: "
                << h_loss
                << std::endl;
        }


        // ====================================================
        // Backward
        // ====================================================

        output_gradient<<<
            batchBlocks,
            threads
        >>>(
            d_Yhat,
            d_Y,
            d_dZ2
        );


        compute_dW2<<<
            hiddenBlocks,
            threads
        >>>(
            d_A1,
            d_dZ2,
            d_dW2
        );


        compute_db2<<<1, 1>>>(
            d_dZ2,
            d_db2
        );


        hidden_gradient<<<
            blocksLayer1,
            threads2D
        >>>(
            d_dZ2,
            d_W2,
            d_Z1,
            d_dZ1
        );


        compute_dW1<<<
            blocksW1,
            threads2D
        >>>(
            d_X,
            d_dZ1,
            d_dW1
        );


        compute_db1<<<
            hiddenBlocks,
            threads
        >>>(
            d_dZ1,
            d_db1
        );


        // ====================================================
        // SGD update
        // ====================================================

        update_parameters<<<
            (w1Elements + threads - 1)
                / threads,
            threads
        >>>(
            d_W1,
            d_dW1,
            w1Elements,
            LEARNING_RATE
        );


        update_parameters<<<
            hiddenBlocks,
            threads
        >>>(
            d_b1,
            d_db1,
            HIDDEN,
            LEARNING_RATE
        );


        update_parameters<<<
            hiddenBlocks,
            threads
        >>>(
            d_W2,
            d_dW2,
            HIDDEN,
            LEARNING_RATE
        );


        update_parameters<<<1, 1>>>(
            d_b2,
            d_db2,
            1,
            LEARNING_RATE
        );


        CUDA_CHECK(
            cudaGetLastError()
        );
    }


    // --------------------------------------------------------
    // Final inference
    // --------------------------------------------------------

    forward_layer1<<<
        blocksLayer1,
        threads2D
    >>>(
        d_X,
        d_W1,
        d_b1,
        d_Z1,
        d_A1
    );


    forward_layer2<<<
        batchBlocks,
        threads
    >>>(
        d_A1,
        d_W2,
        d_b2,
        d_Z2,
        d_Yhat
    );


    float h_Yhat[BATCH];

    CUDA_CHECK(
        cudaMemcpy(
            h_Yhat,
            d_Yhat,
            sizeof(h_Yhat),
            cudaMemcpyDeviceToHost
        )
    );


    std::cout << "\nFinal predictions:\n";

    for (int i = 0; i < BATCH; ++i)
    {
        std::cout
            << h_X[i * INPUT]
            << " XOR "
            << h_X[i * INPUT + 1]
            << " -> "
            << h_Yhat[i]
            << "  class="
            << (h_Yhat[i] >= 0.5f)
            << std::endl;
    }


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    cudaFree(d_X);
    cudaFree(d_Y);

    cudaFree(d_W1);
    cudaFree(d_b1);

    cudaFree(d_W2);
    cudaFree(d_b2);

    cudaFree(d_Z1);
    cudaFree(d_A1);

    cudaFree(d_Z2);
    cudaFree(d_Yhat);

    cudaFree(d_dZ2);
    cudaFree(d_dW2);
    cudaFree(d_db2);

    cudaFree(d_dZ1);
    cudaFree(d_dW1);
    cudaFree(d_db1);

    cudaFree(d_loss);

    return 0;
}
