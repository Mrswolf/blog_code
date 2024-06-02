#include "algo.h"
#include <cstring>

void median(float *pSrc, int N, float &fMedianVal)
{
    if (N < 1)
    {
        return;
    }

    if (N == 1)
    {
        fMedianVal = *pSrc;
        return;
    }

    float *pBuffer = new float[N];
    std::memcpy(pBuffer, pSrc, N * sizeof(float));
    fMedianVal = 0;
    if (N % 2 == 0)
    {
        std::nth_element(pBuffer, pBuffer + N / 2 - 1, pBuffer + N);
        fMedianVal += *(pBuffer + N / 2 - 1);
        std::nth_element(pBuffer, pBuffer + N / 2, pBuffer + N);
        fMedianVal += *(pBuffer + N / 2);
        fMedianVal /= 2;
    }
    else
    {
        std::nth_element(pBuffer, pBuffer + N / 2, pBuffer + N);
        fMedianVal += *(pBuffer + N / 2);
    }
    delete[] pBuffer;
}