#include "cuda_runtime.h"

#if defined _WIN32 || defined __CYGWIN__
#define HELPER_DLL_IMPORT __declspec(dllimport)
#define HELPER_DLL_EXPORT __declspec(dllexport)
#define HELPER_DLL_LOCAL
#else
#if __GNUC__ >= 4
#define HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define HELPER_DLL_IMPORT
#define HELPER_DLL_EXPORT
#define HELPER_DLL_LOCAL
#endif
#endif

#ifdef EXPORT_ALL
#define DLL_API HELPER_DLL_EXPORT
#else
#define DLL_API HELPER_DLL_IMPORT
#endif

DLL_API void hello();

void hello(int i);

__global__ void cudahello();

DLL_API void cudaHelloLaunch();