#include <iostream>
#include <iomanip>
#include <memory>
#include <cstring>
#include <numeric>
#include <chrono>
#include <vector>
#include <omp.h>

template <typename T>
void displayMat(T *pSrcDst, int M, int N)
{
    std::cout << "======" << std::endl;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << std::setw(4) << pSrcDst[i * N + j];
        }
        std::cout << std::endl;
    }
    std::cout << "======" << std::endl;
}

template <typename T>
void reverse_with_stride(T *pSrcDst, size_t N, size_t stride)
{
    if (N <= 1)
    {
        return;
    }
    T *pa, *pb;
    pa = pSrcDst;
    pb = pSrcDst + N * stride;

    N /= 2;
    for (int64_t i = 0; i < N; ++i)
    {
        T buffer(*(pa + i * stride));
        *(pa + i * stride) = *(pb - (i + 1) * stride);
        *(pb - (i + 1) * stride) = buffer;
    }
}

template <typename T>
void triple_reversal_rotation(T *pSrcDst, size_t left, size_t right, size_t stride)
{
    if (left == 0 || right == 0)
    {
        return;
    }
    reverse_with_stride(pSrcDst, left, stride);
    reverse_with_stride(pSrcDst + left * stride, right, stride);
    reverse_with_stride(pSrcDst, left + right, stride);
}

template <typename T>
void circshift(const std::vector<int> &vDims, int iAxis, int shift, T *pSrcDst)
{
    // modulo the shift in positive range
    shift = (shift >= 0) ? shift % vDims[iAxis] : vDims[iAxis] + shift % vDims[iAxis];
    size_t stride = 1;
    for (size_t i = iAxis + 1; i < vDims.size(); ++i)
    {
        stride *= vDims[i];
    }
    size_t left = vDims[iAxis] - shift;
    size_t right = shift;
    size_t lOuterLoops = 1;
    for (size_t i = 0; i < iAxis; ++i)
    {
        lOuterLoops *= vDims[i];
    }
    int64_t lOuterStride = stride * vDims[iAxis];

    size_t lInnerLoops = stride;
    int64_t lInnerStride = 1;
    // std::cout << "shift:" << shift << " stride:" << stride << " left:" << left << " right:" << right << std::endl;
    // std::cout << "lOuterLoops:" << lOuterLoops << " lOuterStride:" << lOuterStride << " lInnerLoops:" << lInnerLoops << " lInnerStride:" << lInnerStride << std::endl;

    int iMaxThreads = omp_get_max_threads();
    int iNumThreads = iMaxThreads < (lOuterLoops * lInnerLoops) ? iMaxThreads : lOuterLoops * lInnerLoops;
#pragma omp parallel for collapse(2) num_threads(iNumThreads) schedule(static) if (iNumThreads > 1)
    for (int64_t i = 0; i < lOuterLoops; ++i)
    {
        for (int64_t j = 0; j < lInnerLoops; ++j)
        {
            // std::cout << "Thread id:" << tid << std::endl;
            triple_reversal_rotation(pSrcDst + i * lOuterStride + j * lInnerStride, left, right, stride);
        }
    }
}

int main(int argc, char *argv[])
{
    std::cout << "Hello, from circshift!\n";

    std::vector<int> vDims = {3, 4, 5};
    int iSize = std::accumulate(vDims.begin(), vDims.end(), 1, std::multiplies<int>());
    std::shared_ptr<int> spData(new int[iSize]);
    int *pData = spData.get();

    for (int i = 0; i < iSize; ++i)
    {
        pData[i] = i;
    }

    int iStride = vDims[1] * vDims[2];

    for (int i = 0; i < vDims[0]; ++i)
    {
        displayMat(pData + i * iStride, vDims[1], vDims[2]);
    }

    std::cout << "circshift along the 2nd dimension" << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();
    circshift(vDims, 1, 100, pData);
    const auto end = std::chrono::high_resolution_clock::now();
    int64_t lDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Elapsed times: " << lDuration << "us" << std::endl;

    for (int i = 0; i < vDims[0]; ++i)
    {
        displayMat(pData + i * iStride, vDims[1], vDims[2]);
    }
}
