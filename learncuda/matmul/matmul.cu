#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_KERNEL_TIME(val, ms) std::cout << #val " Kernel execution time: " << ms << "ms" << std::endl

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

void CudaDeviceInfo()
{
    int iDeviceId;
    cudaGetDevice(&iDeviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, iDeviceId);
    printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
           iDeviceId, props.name, props.major, props.minor, props.memoryBusWidth,
           props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
           props.regsPerBlock, props.regsPerMultiprocessor,
           props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
           props.multiProcessorCount, props.warpSize);
}

void check_val_error(float *A, float *B, int M, int N)
{
    float temp = 0.0f;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            temp += (A[i * N + j] - B[i * N + j]);
        }
    }
    std::cout << "error: " << temp << std::endl;
}

void MySaveBin(std::string sFileName, void *pData, size_t size)
{
    std::ofstream outFile(sFileName, std::ofstream::binary);
    outFile.write((char *)pData, size);
    outFile.close();
}

__global__ void sgemm_naive(
    int M, int N, int K, const float alpha, const float *A, const float *B, const float beta, float *C)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    // avoids memory access error if threads are more than elements
    if (i < M && j < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
}

__global__ void sgemm_coalesce(
    int M, int N, int K, const float alpha, const float *A, const float *B, const float beta, float *C)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    // avoids memory access error if threads are more than elements
    if (i < M && j < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
}

#define BLOCKSIZE 32

__global__ void sgemm_shared(
    int M, int N, int K, const float alpha, const float *A, const float *B, const float beta, float *C)
{
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    A += blockIdx.y * BLOCKSIZE * K;
    B += blockIdx.x * BLOCKSIZE;
    C += blockIdx.y * BLOCKSIZE * N + blockIdx.x * BLOCKSIZE;

    const int j = blockIdx.x * BLOCKSIZE + threadIdx.x;
    const int i = blockIdx.y * BLOCKSIZE + threadIdx.y;

    // avoids memory access error if threads are more than elements
    if (i < M && j < N)
    {
        float fSum = 0.0f; // stores result of (threadIdx.y, threadIdx.x) on each block
        for (int iBlkIdx = 0; iBlkIdx < K; iBlkIdx += BLOCKSIZE)
        {
            if (iBlkIdx + threadIdx.x < K)
            {
                As[threadIdx.y * BLOCKSIZE + threadIdx.x] = A[threadIdx.y * K + threadIdx.x];
            }
            if (iBlkIdx + threadIdx.y < K)
            {
                Bs[threadIdx.y * BLOCKSIZE + threadIdx.x] = B[threadIdx.y * N + threadIdx.x];
            }
            __syncthreads(); // syncronize until  all caches are fulfilled

            // updates to the next chunk
            A += BLOCKSIZE;
            B += BLOCKSIZE * N;

            // dot product on caches
            for (int iInnerLoop = 0; iInnerLoop < BLOCKSIZE; ++iInnerLoop)
            {
                if (iBlkIdx + iInnerLoop < K)
                {
                    fSum += As[threadIdx.y * BLOCKSIZE + iInnerLoop] * Bs[iInnerLoop * BLOCKSIZE + threadIdx.x];
                }
            }

            __syncthreads();
        }

        C[threadIdx.y * N + threadIdx.x] = alpha * fSum + beta * C[threadIdx.y * N + threadIdx.x];
    }
}
template <typename T>
void InitRandomArray(T *pSrcDst, int N)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1, 1);
    for (int i = 0; i < N; ++i)
    {
        pSrcDst[i] = dis(gen);
    }
}

int main()
{
    int M(1024), K(1024), N(1024);
    M = N = K = 4096;
    float alpha(1.0f), beta(0.0f);
    std::shared_ptr<float> spA(new float[M * K]);
    std::shared_ptr<float> spB(new float[K * N]);
    std::shared_ptr<float> spC(new float[M * N]);
    std::shared_ptr<float> spGroundTruthC(new float[M * N]);

    CudaDeviceInfo();

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    InitRandomArray(spA.get(), M * K);
    InitRandomArray(spB.get(), K * N);
    InitRandomArray(spC.get(), M * N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cublasStatus_t stat;

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, spA.get(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, spB.get(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, spC.get(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    CHECK_CUDA_ERROR(cudaMemset(d_C, 0, M * N * sizeof(float)));

    // cublas
    cudaEventRecord(start);
    // cublas uses column-major, thus we compute (AB)^T instead of AB
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaEventRecord(stop);
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Kernel execution time: " << milliseconds << "ms" << std::endl;
    CUDA_KERNEL_TIME(cublasSgemm, milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(spGroundTruthC.get(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // naive
    CHECK_CUDA_ERROR(cudaMemset(d_C, 0, M * N * sizeof(float)));
    dim3 gridDim((M - 1) / 32 + 1, (N - 1) / 32 + 1);
    dim3 blockDim(32, 32, 1);
    cudaEventRecord(start);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(stop);
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);
    CUDA_KERNEL_TIME(sgemm_naive, milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(spC.get(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    check_val_error(spC.get(), spGroundTruthC.get(), M, N);

    // coalesce
    CHECK_CUDA_ERROR(cudaMemset(d_C, 0, M * N * sizeof(float)));
    gridDim = {(N - 1) / 32 + 1, (M - 1) / 32 + 1, 1};
    cudaEventRecord(start);
    sgemm_coalesce<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(stop);
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);
    CUDA_KERNEL_TIME(sgemm_coalesce, milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(spC.get(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    check_val_error(spC.get(), spGroundTruthC.get(), M, N);

    // shared
    CHECK_CUDA_ERROR(cudaMemset(d_C, 0, M * N * sizeof(float)));
    blockDim = {32, 32, 1};
    gridDim = {(N - 1) / 32 + 1, (M - 1) / 32 + 1, 1};
    cudaEventRecord(start);
    sgemm_shared<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(stop);
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);
    CUDA_KERNEL_TIME(sgemm_shared, milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(spC.get(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    check_val_error(spC.get(), spGroundTruthC.get(), M, N);

    // MySaveBin("groudtruthC.bin", spGroundTruthC.get(), M * N * sizeof(float));
    // MySaveBin("C.bin", spC.get(), M * N * sizeof(float));

    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    cublasDestroy(handle);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    return 0;
}