#include <iostream>
#include <vector>
#include "algo.h"
#include "stalib/stalib.h"
#include "dynlib/dynlib.h"

int main(int argc, char *argv[])
{
    std::cout << "HELLO CMAKE!" << std::endl;
    hello();
    cudaHelloLaunch();

    std::vector<float>
        vVals = {1, 3, 4, 4, 5, 2, 2, 3, (float)getNum()};
    vVals.push_back(myrand());
    vVals.push_back(myrand());

    float fMedianVal(0.0);
    median(vVals.data(), vVals.size(), fMedianVal);
    std::cout << "Median is " << fMedianVal << std::endl;
    return 0;
}