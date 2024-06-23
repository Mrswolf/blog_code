#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 10000

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

__global__ void addvec(int *a, int *b, int *c)
{
    int iThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int iStride = blockDim.x * gridDim.x;
    for (int i = iThreadIdx; i < N; i += iStride)
    {
        c[i] = a[i] + b[i];
    }
}

void initArray(int *p, int value)
{
    for (int i = 0; i < N; ++i)
    {
        p[i] = value;
    }
}

int main()
{
    int *a, *b, *c;
    a = (int *)malloc(N * sizeof(int));
    b = (int *)malloc(N * sizeof(int));
    c = (int *)malloc(N * sizeof(int));

    initArray(a, 1);
    initArray(b, 2);
    initArray(c, 0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *da, *db, *dc;
    CHECK_CUDA_ERROR(cudaMalloc(&da, N * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&db, N * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&dc, N * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(da, a, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(db, b, N * sizeof(int), cudaMemcpyHostToDevice));

    int iThreads = 32;
    int iBlocks = (N - 1) / iThreads + 1;
    std::cout << "Threads: " << iThreads << " Blocks: " << iBlocks << std::endl;

    // auto start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);
    addvec<<<iBlocks, iThreads>>>(da, db, dc);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "Kernel execution time: " << duration.count() << "us" << std::endl;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << "ms" << std::endl;

    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaMemcpy(c, dc, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; ++i)
    {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);

    return 0;
}