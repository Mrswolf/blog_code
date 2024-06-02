#include "dynlib.h"
#include <iostream>

void hello()
{
    std::cout << "hello dynlib" << std::endl;
}

void hello2(int i)
{

    std::cout << "hello" << i << std::endl;
}