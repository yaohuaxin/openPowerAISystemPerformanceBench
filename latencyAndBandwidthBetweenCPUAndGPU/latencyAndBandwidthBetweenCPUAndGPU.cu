/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This is a simple test program to measure the memcopy bandwidth of the GPU.
 * It can measure device to device copy bandwidth, host to device copy bandwidth
 * for pageable and pinned memory, and device to host copy bandwidth for pageable
 * and pinned memory.
 *
 * Usage:
 * ./bandwidthTest [option]...
 */

// CUDA runtime
#include <cuda_runtime.h>

// includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

#include <cuda.h>

#include <memory>
#include <iostream>
#include <cassert>

#include <cstdio>
#include <vector>

static const char *sSDKsample = "Latency and Bandwidth Test";

// defines, project
#define MEMCOPY_ITERATIONS  100
#define MEMCOPY_ITERATIONS_For_Test_Latency  100
#define DEFAULT_SIZE        ( 32 * ( 1 << 20 ) )    //32 M
#define DEFAULT_SIZE_For_Test_Latency        ( 1 )
#define DEFAULT_INCREMENT   (1 << 22)               //4 M
#define CACHE_CLEAR_SIZE    (1 << 24)               //16 M

//shmoo mode defines
#define SHMOO_MEMSIZE_MAX     (1 << 26)         //64 M
#define SHMOO_MEMSIZE_START   (1 << 10)         //1 KB
#define SHMOO_INCREMENT_1KB   (1 << 10)         //1 KB
#define SHMOO_INCREMENT_2KB   (1 << 11)         //2 KB
#define SHMOO_INCREMENT_10KB  (10 * (1 << 10))  //10KB
#define SHMOO_INCREMENT_100KB (100 * (1 << 10)) //100 KB
#define SHMOO_INCREMENT_1MB   (1 << 20)         //1 MB
#define SHMOO_INCREMENT_2MB   (1 << 21)         //2 MB
#define SHMOO_INCREMENT_4MB   (1 << 22)         //4 MB
#define SHMOO_LIMIT_20KB      (20 * (1 << 10))  //20 KB
#define SHMOO_LIMIT_50KB      (50 * (1 << 10))  //50 KB
#define SHMOO_LIMIT_100KB     (100 * (1 << 10)) //100 KB
#define SHMOO_LIMIT_1MB       (1 << 20)         //1 MB
#define SHMOO_LIMIT_16MB      (1 << 24)         //16 MB
#define SHMOO_LIMIT_32MB      (1 << 25)         //32 MB

//enums, project
enum testMode   { QUICK_MODE, RANGE_MODE, SHMOO_MODE };
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
enum printMode  { USER_READABLE, CSV };
enum memoryMode { PINNED, PAGEABLE };

const char *sMemoryCopyKind[] =
{
    "Device to Host",
    "Host to Device",
    "Device to Device",
    NULL
};

const char *sMemoryMode[] =
{
    "PINNED",
    "PAGEABLE",
    NULL
};

// if true, use CPU based timing for everything
static bool bDontUseGPUTiming;

int *pArgc = NULL;
char **pArgv = NULL;

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runBandwidthTest(const int argc, const char **argv);
void testBandwidth(unsigned int start, unsigned int end, unsigned int increment,
                   testMode mode, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
void testBandwidthQuick(unsigned int size, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
void testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment,
                        memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
void testBandwidthShmoo(memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc);
float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc);
float testDeviceToDeviceTransfer(unsigned int memSize);
void outputLatencyMatrix(int numGPUs, bool p2p, memcpyKind kind);
void printResultsReadable(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, int iNumDevs, bool wc);
void printResultsCSV(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, int iNumDevs, bool wc);
void printHelp(void);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    // set logfile name and start logs
    printf("[%s] - Starting...\n", sSDKsample);

    int iRetVal = runBandwidthTest(argc, (const char **)argv);

    if (iRetVal < 0)
    {
        checkCudaErrors(cudaSetDevice(0));
    }

    // latency tests
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }
    printf("\nHOST_TO_DEVICE \n");
    outputLatencyMatrix(deviceCount, true, HOST_TO_DEVICE);

    printf("\nDEVICE_TO_HOST \n");
    outputLatencyMatrix(deviceCount, true, DEVICE_TO_HOST);

    printf("\nDEVICE_TO_DEVICE \n");
    outputLatencyMatrix(deviceCount, true, DEVICE_TO_DEVICE);

    // finish
    printf("%s\n", (iRetVal==0) ? "Result = PASS" : "Result = FAIL");

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    exit((iRetVal==0) ? EXIT_SUCCESS : EXIT_FAILURE);
}

///////////////////////////////////////////////////////////////////////////////
//Parse args, run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
int runBandwidthTest(const int argc, const char **argv)
{
    int start = DEFAULT_SIZE;
    int end = DEFAULT_SIZE;
    int startDevice = 0;
    int endDevice = 0;
    int increment = DEFAULT_INCREMENT;
    testMode mode = QUICK_MODE;
    bool htod = false;
    bool dtoh = false;
    bool dtod = false;
    bool wc = false;
    char *modeStr;
    char *device = NULL;
    printMode printmode = USER_READABLE;
    char *memModeStr = NULL;
    memoryMode memMode = PINNED;

    //process command line args
    if (checkCmdLineFlag(argc, argv, "help"))
    {
        printHelp();
        return 0;
    }

    if (checkCmdLineFlag(argc, argv, "csv"))
    {
        printmode = CSV;
    }

    if (getCmdLineArgumentString(argc, argv, "memory", &memModeStr))
    {
        if (strcmp(memModeStr, "pageable") == 0)
        {
            memMode = PAGEABLE;
        }
        else if (strcmp(memModeStr, "pinned") == 0)
        {
            memMode = PINNED;
        }
        else
        {
            printf("Invalid memory mode - valid modes are pageable or pinned\n");
            printf("See --help for more information\n");
            return -1000;
        }
    }
    else
    {
        //default - pinned memory
        memMode = PINNED;
    }

    if (getCmdLineArgumentString(argc, argv, "device", &device))
    {
        int deviceCount;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

        if (error_id != cudaSuccess)
        {
            printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
            exit(EXIT_FAILURE);
        }

        if (deviceCount == 0)
        {
            printf("!!!!!No devices found!!!!!\n");
            return -2000;
        }

        if (strcmp(device, "all") == 0)
        {
            printf("\n!!!!!Cumulative Bandwidth to be computed from all the devices !!!!!!\n\n");
            startDevice = 0;
            endDevice = deviceCount-1;
        }
        else
        {
            startDevice = endDevice = atoi(device);

            if (startDevice >= deviceCount || startDevice < 0)
            {
                printf("\n!!!!!Invalid GPU number %d given hence default gpu %d will be used !!!!!\n", startDevice,0);
                startDevice = endDevice = 0;
            }
        }
    }
    else
    {
    	printf("Device No is not set, use the default device: 0.\n");
    }

    printf("Running Bandwidth ...\n\n");

    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        cudaDeviceProp deviceProp;
        cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, currentDevice);

        if (error_id == cudaSuccess)
        {
            printf(" Device %d: %s\n", currentDevice, deviceProp.name);

            if (deviceProp.computeMode == cudaComputeModeProhibited)
            {
                fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
                checkCudaErrors(cudaSetDevice(currentDevice));

                exit(EXIT_FAILURE);
            }
        }
        else
        {
            printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
            checkCudaErrors(cudaSetDevice(currentDevice));

            exit(EXIT_FAILURE);
        }
    }

    if (getCmdLineArgumentString(argc, argv, "mode", &modeStr))
    {
        //figure out the mode
        if (strcmp(modeStr, "quick") == 0)
        {
            printf(" Quick Mode\n\n");
            mode = QUICK_MODE;
        }
        else if (strcmp(modeStr, "shmoo") == 0)
        {
            printf(" Shmoo Mode\n\n");
            mode = SHMOO_MODE;
        }
        else if (strcmp(modeStr, "range") == 0)
        {
            printf(" Range Mode\n\n");
            mode = RANGE_MODE;
        }
        else
        {
            printf("Invalid mode - valid modes are quick, range, or shmoo\n");
            printf("See --help for more information\n");
            return -3000;
        }
    }
    else
    {
        //default mode - quick
        printf(" Quick Mode\n\n");
        mode = QUICK_MODE;
    }

    if (checkCmdLineFlag(argc, argv, "htod"))
    {
        htod = true;
    }

    if (checkCmdLineFlag(argc, argv, "dtoh"))
    {
        dtoh = true;
    }

    if (checkCmdLineFlag(argc, argv, "dtod"))
    {
        dtod = true;
    }

#if CUDART_VERSION >= 2020

    if (checkCmdLineFlag(argc, argv, "wc"))
    {
        wc = true;
    }

#endif

    if (checkCmdLineFlag(argc, argv, "cputiming"))
    {
        bDontUseGPUTiming = true;
    }

    if (!htod && !dtoh && !dtod)
    {
        //default:  All
        htod = true;
        dtoh = true;
        dtod = true;
    }

    if (RANGE_MODE == mode)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "start"))
        {
            start = getCmdLineArgumentInt(argc, argv, "start");

            if (start <= 0)
            {
                printf("Illegal argument - start must be greater than zero\n");
                return -4000;
            }
        }
        else
        {
            printf("Must specify a starting size in range mode\n");
            printf("See --help for more information\n");
            return -5000;
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "end"))
        {
            end = getCmdLineArgumentInt(argc, argv, "end");

            if (end <= 0)
            {
                printf("Illegal argument - end must be greater than zero\n");
                return -6000;
            }

            if (start > end)
            {
                printf("Illegal argument - start is greater than end\n");
                return -7000;
            }
        }
        else
        {
            printf("Must specify an end size in range mode.\n");
            printf("See --help for more information\n");
            return -8000;
        }


        if (checkCmdLineFlag(argc, argv, "increment"))
        {
            increment = getCmdLineArgumentInt(argc, argv, "increment");

            if (increment <= 0)
            {
                printf("Illegal argument - increment must be greater than zero\n");
                return -9000;
            }
        }
        else
        {
            printf("Must specify an increment in user mode\n");
            printf("See --help for more information\n");
            return -10000;
        }
    }

    if (htod)
    {
        testBandwidth((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                      mode, HOST_TO_DEVICE, printmode, memMode, startDevice, endDevice, wc);
    }

    if (dtoh)
    {
        testBandwidth((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                      mode, DEVICE_TO_HOST, printmode, memMode, startDevice, endDevice, wc);
    }

    if (dtod)
    {
        testBandwidth((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                      mode, DEVICE_TO_DEVICE, printmode, memMode, startDevice, endDevice, wc);
    }

    // Ensure that we reset all CUDA Devices in question
    for (int nDevice = startDevice; nDevice <= endDevice; nDevice++)
    {
        cudaSetDevice(nDevice);
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
//  Run a bandwidth test
///////////////////////////////////////////////////////////////////////////////
void
testBandwidth(unsigned int start, unsigned int end, unsigned int increment,
              testMode mode, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc)
{
    switch (mode)
    {
        case QUICK_MODE:
            testBandwidthQuick(DEFAULT_SIZE, kind, printmode, memMode, startDevice, endDevice, wc);
            break;

        case RANGE_MODE:
            testBandwidthRange(start, end, increment, kind, printmode, memMode, startDevice, endDevice, wc);
            break;

        case SHMOO_MODE:
            testBandwidthShmoo(kind, printmode, memMode, startDevice, endDevice, wc);
            break;

        default:
            break;
    }
}

//////////////////////////////////////////////////////////////////////
//  Run a quick mode bandwidth test
//////////////////////////////////////////////////////////////////////
void
testBandwidthQuick(unsigned int size, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc)
{
    testBandwidthRange(size, size, DEFAULT_INCREMENT, kind, printmode, memMode, startDevice, endDevice, wc);
}

///////////////////////////////////////////////////////////////////////
//  Run a range mode bandwidth test
//////////////////////////////////////////////////////////////////////
void
testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment,
                   memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc)
{
    //count the number of copies we're going to run
    unsigned int count = 1 + ((end - start) / increment);

    unsigned int *memSizes = (unsigned int *)malloc(count * sizeof(unsigned int));
    double *bandwidths = (double *) malloc(count * sizeof(double));

    // Before calculating the cumulative bandwidth, initialize bandwidths array to NULL
    for (unsigned int i = 0; i < count; i++)
    {
        bandwidths[i] = 0.0;
    }

    // Use the device asked by the user
    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        cudaSetDevice(currentDevice);

        //run each of the copies
        for (unsigned int i = 0; i < count; i++)
        {

            memSizes[i] = start + i * increment;

            switch (kind)
            {
                case DEVICE_TO_HOST:
                    bandwidths[i] += testDeviceToHostTransfer(memSizes[i], memMode, wc);
                    break;

                case HOST_TO_DEVICE:
                    bandwidths[i] += testHostToDeviceTransfer(memSizes[i], memMode, wc);
                    break;

                case DEVICE_TO_DEVICE:
                    bandwidths[i] += testDeviceToDeviceTransfer(memSizes[i]);
                    break;
            }
        }
    } // Complete the bandwidth computation on all the devices

    //print results
    if (printmode == CSV)
    {
        printResultsCSV(memSizes, bandwidths, count, kind, memMode, (1 + endDevice - startDevice), wc);
    }
    else
    {
        printResultsReadable(memSizes, bandwidths, count, kind, memMode, (1 + endDevice - startDevice), wc);
    }

    //clean up
    free(memSizes);
    free(bandwidths);
}

//////////////////////////////////////////////////////////////////////////////
// Intense shmoo mode - covers a large range of values with varying increments
//////////////////////////////////////////////////////////////////////////////
void
testBandwidthShmoo(memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc)
{
    //count the number of copies to make
    unsigned int count = 1 + (SHMOO_LIMIT_20KB  / SHMOO_INCREMENT_1KB)
                         + ((SHMOO_LIMIT_50KB - SHMOO_LIMIT_20KB) / SHMOO_INCREMENT_2KB)
                         + ((SHMOO_LIMIT_100KB - SHMOO_LIMIT_50KB) / SHMOO_INCREMENT_10KB)
                         + ((SHMOO_LIMIT_1MB - SHMOO_LIMIT_100KB) / SHMOO_INCREMENT_100KB)
                         + ((SHMOO_LIMIT_16MB - SHMOO_LIMIT_1MB) / SHMOO_INCREMENT_1MB)
                         + ((SHMOO_LIMIT_32MB - SHMOO_LIMIT_16MB) / SHMOO_INCREMENT_2MB)
                         + ((SHMOO_MEMSIZE_MAX - SHMOO_LIMIT_32MB) / SHMOO_INCREMENT_4MB);

    unsigned int *memSizes = (unsigned int *)malloc(count * sizeof(unsigned int));
    double *bandwidths = (double *) malloc(count * sizeof(double));

    // Before calculating the cumulative bandwidth, initialize bandwidths array to NULL
    for (unsigned int i = 0; i < count; i++)
    {
        bandwidths[i] = 0.0;
    }

    // Use the device asked by the user
    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        cudaSetDevice(currentDevice);
        //Run the shmoo
        int iteration = 0;
        unsigned int memSize = 0;

        while (memSize <= SHMOO_MEMSIZE_MAX)
        {
            if (memSize < SHMOO_LIMIT_20KB)
            {
                memSize += SHMOO_INCREMENT_1KB;
            }
            else if (memSize < SHMOO_LIMIT_50KB)
            {
                memSize += SHMOO_INCREMENT_2KB;
            }
            else if (memSize < SHMOO_LIMIT_100KB)
            {
                memSize += SHMOO_INCREMENT_10KB;
            }
            else if (memSize < SHMOO_LIMIT_1MB)
            {
                memSize += SHMOO_INCREMENT_100KB;
            }
            else if (memSize < SHMOO_LIMIT_16MB)
            {
                memSize += SHMOO_INCREMENT_1MB;
            }
            else if (memSize < SHMOO_LIMIT_32MB)
            {
                memSize += SHMOO_INCREMENT_2MB;
            }
            else
            {
                memSize += SHMOO_INCREMENT_4MB;
            }

            memSizes[iteration] = memSize;

            switch (kind)
            {
                case DEVICE_TO_HOST:
                    bandwidths[iteration] += testDeviceToHostTransfer(memSizes[iteration], memMode, wc);
                    break;

                case HOST_TO_DEVICE:
                    bandwidths[iteration] += testHostToDeviceTransfer(memSizes[iteration], memMode, wc);
                    break;

                case DEVICE_TO_DEVICE:
                    bandwidths[iteration] += testDeviceToDeviceTransfer(memSizes[iteration]);
                    break;
            }

            iteration++;
            printf(".");
        }
    } // Complete the bandwidth computation on all the devices

    //print results
    printf("\n");

    if (CSV == printmode)
    {
        printResultsCSV(memSizes, bandwidths, count, kind, memMode, (1 + endDevice - startDevice), wc);
    }
    else
    {
        printResultsReadable(memSizes, bandwidths, count, kind, memMode, (1 + endDevice - startDevice), wc);
    }

    //clean up
    free(memSizes);
    free(bandwidths);
}

///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float
testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    unsigned char *h_idata = NULL;
    unsigned char *h_odata = NULL;
    cudaEvent_t start, stop;

    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    //allocate host memory
    if (PINNED == memMode)
    {
        //pinned memory mode - use special function to get OS-pinned memory
#if CUDART_VERSION >= 2020
        checkCudaErrors(cudaHostAlloc((void **)&h_idata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
        checkCudaErrors(cudaHostAlloc((void **)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
#else
        checkCudaErrors(cudaMallocHost((void **)&h_idata, memSize));
        checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
#endif
    }
    else
    {
        //pageable memory mode - use malloc
        h_idata = (unsigned char *)malloc(memSize);
        h_odata = (unsigned char *)malloc(memSize);

        if (h_idata == 0 || h_odata == 0)
        {
            fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
            exit(EXIT_FAILURE);
        }
    }

    //initialize the memory
    for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_idata[i] = (unsigned char)(i & 0xff);
    }

    // allocate device memory
    unsigned char *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, memSize));

    //initialize the device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize,
                               cudaMemcpyHostToDevice));

    //copy data from GPU to Host
    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

    if (PINNED == memMode)
    {
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            checkCudaErrors(cudaMemcpyAsync(h_odata, d_idata, memSize,
                                            cudaMemcpyDeviceToHost, 0));
        }
    }
    else
    {
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            checkCudaErrors(cudaMemcpy(h_odata, d_idata, memSize,
                                       cudaMemcpyDeviceToHost));
        }
    }

    checkCudaErrors(cudaEventRecord(stop, 0));

    // make sure GPU has finished copying
    checkCudaErrors(cudaDeviceSynchronize());
    //get the total elapsed time in ms
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    if (PINNED != memMode || bDontUseGPUTiming)
    {
        elapsedTimeInMs = sdkGetTimerValue(&timer);
    }

    //calculate bandwidth in MB/s
    bandwidthInMBs = ((float)(1<<10) * memSize * (float)MEMCOPY_ITERATIONS) /
                     (elapsedTimeInMs * (float)(1 << 20));

    //clean up memory
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    sdkDeleteTimer(&timer);

    if (PINNED == memMode)
    {
        checkCudaErrors(cudaFreeHost(h_idata));
        checkCudaErrors(cudaFreeHost(h_odata));
    }
    else
    {
        free(h_idata);
        free(h_odata);
    }

    checkCudaErrors(cudaFree(d_idata));

    return bandwidthInMBs;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a host to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float
testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cudaEvent_t start, stop;
    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    //allocate host memory
    unsigned char *h_odata = NULL;

    if (PINNED == memMode)
    {
#if CUDART_VERSION >= 2020
        //pinned memory mode - use special function to get OS-pinned memory
        checkCudaErrors(cudaHostAlloc((void **)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
#else
        //pinned memory mode - use special function to get OS-pinned memory
        checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
#endif
    }
    else
    {
        //pageable memory mode - use malloc
        h_odata = (unsigned char *)malloc(memSize);

        if (h_odata == 0)
        {
            fprintf(stderr, "Not enough memory available on host to run test!\n");
            exit(EXIT_FAILURE);
        }
    }

    unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
    unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);

    if (h_cacheClear1 == 0 || h_cacheClear2 == 0)
    {
        fprintf(stderr, "Not enough memory available on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    //initialize the memory
    for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_odata[i] = (unsigned char)(i & 0xff);
    }

    for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++)
    {
        h_cacheClear1[i] = (unsigned char)(i & 0xff);
        h_cacheClear2[i] = (unsigned char)(0xff - (i & 0xff));
    }

    //allocate device memory
    unsigned char *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, memSize));

    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

    //copy host memory to device memory
    if (PINNED == memMode)
    {
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            checkCudaErrors(cudaMemcpyAsync(d_idata, h_odata, memSize,
                                            cudaMemcpyHostToDevice, 0));
        }
    }
    else
    {
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            checkCudaErrors(cudaMemcpy(d_idata, h_odata, memSize,
                                       cudaMemcpyHostToDevice));
        }
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    //total elapsed time in ms
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    if (PINNED != memMode || bDontUseGPUTiming)
    {
        elapsedTimeInMs = sdkGetTimerValue(&timer);
    }

    sdkResetTimer(&timer);

    //calculate bandwidth in MB/s
    bandwidthInMBs = ((float)(1<<10) * memSize * (float)MEMCOPY_ITERATIONS) /
                     (elapsedTimeInMs * (float)(1 << 20));

    //clean up memory
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    sdkDeleteTimer(&timer);

    if (PINNED == memMode)
    {
        checkCudaErrors(cudaFreeHost(h_odata));
    }
    else
    {
        free(h_odata);
    }

    free(h_cacheClear1);
    free(h_cacheClear2);
    checkCudaErrors(cudaFree(d_idata));

    return bandwidthInMBs;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a device to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float
testDeviceToDeviceTransfer(unsigned int memSize)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cudaEvent_t start, stop;

    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    //allocate host memory
    unsigned char *h_idata = (unsigned char *)malloc(memSize);

    if (h_idata == 0)
    {
        fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    //initialize the host memory
    for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_idata[i] = (unsigned char)(i & 0xff);
    }

    //allocate device memory
    unsigned char *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, memSize));
    unsigned char *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, memSize));

    //initialize memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize,
                               cudaMemcpyHostToDevice));

    //run the memcopy
    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        checkCudaErrors(cudaMemcpy(d_odata, d_idata, memSize,
                                   cudaMemcpyDeviceToDevice));
    }

    checkCudaErrors(cudaEventRecord(stop, 0));

    //Since device to device memory copies are non-blocking,
    //cudaDeviceSynchronize() is required in order to get
    //proper timing.
    checkCudaErrors(cudaDeviceSynchronize());

    //get the total elapsed time in ms
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    if (bDontUseGPUTiming)
    {
        elapsedTimeInMs = sdkGetTimerValue(&timer);
    }

    //calculate bandwidth in MB/s
    bandwidthInMBs = 2.0f * ((float)(1<<10) * memSize * (float)MEMCOPY_ITERATIONS) /
                     (elapsedTimeInMs * (float)(1 << 20));

    //clean up memory
    sdkDeleteTimer(&timer);
    free(h_idata);
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    return bandwidthInMBs;
}

/////////////////////////////////////////////////////////
//print results in an easily read format
////////////////////////////////////////////////////////
void printResultsReadable(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, int iNumDevs, bool wc)
{
    printf(" %s Bandwidth, %i Device(s)\n", sMemoryCopyKind[kind], iNumDevs);
    printf(" %s Memory Transfers\n", sMemoryMode[memMode]);

    if (wc)
    {
        printf(" Write-Combined Memory Writes are Enabled");
    }

    printf("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");
    unsigned int i;

    for (i = 0; i < (count - 1); i++)
    {
        printf("   %u\t\t\t%s%.1f\n", memSizes[i], (memSizes[i] < 10000)? "\t" : "", bandwidths[i]);
    }

    printf("   %u\t\t\t%s%.1f\n\n", memSizes[i], (memSizes[i] < 10000)? "\t" : "", bandwidths[i]);
}

///////////////////////////////////////////////////////////////////////////
//print results in a database format
///////////////////////////////////////////////////////////////////////////
void printResultsCSV(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, int iNumDevs, bool wc)
{
    std::string sConfig;

    // log config information
    if (kind == DEVICE_TO_DEVICE)
    {
        sConfig += "D2D";
    }
    else
    {
        if (kind == DEVICE_TO_HOST)
        {
            sConfig += "D2H";
        }
        else if (kind == HOST_TO_DEVICE)
        {
            sConfig += "H2D";
        }

        if (memMode == PAGEABLE)
        {
            sConfig += "-Paged";
        }
        else if (memMode == PINNED)
        {
            sConfig += "-Pinned";

            if (wc)
            {
                sConfig += "-WriteCombined";
            }
        }
    }

    unsigned int i;
    double dSeconds = 0.0;

    for (i = 0; i < count; i++)
    {
        dSeconds = (double)memSizes[i] / (bandwidths[i] * (double)(1<<20));
        printf("bandwidthTest-%s, Bandwidth = %.1f MB/s, Time = %.5f s, Size = %u bytes, NumDevsUsed = %d\n",
               sConfig.c_str(), bandwidths[i], dSeconds, memSizes[i], iNumDevs);
    }
}



__global__ void delay(volatile int *flag, unsigned long long timeout_clocks = 100000000)
{
    // Wait until the application notifies us that it has completed queuing up the
    // experiment, or timeout and exit, allowing the application to make progress
    long long int start_clock, sample_clock;
    start_clock = clock64();

    while (!*flag) {
        sample_clock = clock64();

        if (sample_clock - start_clock > timeout_clocks) {
            break;
        }
    }
}

void performHostToDeviceCopy(int *dest, int *src, int num_elems, int repeat, cudaStream_t streamToRun)
{
	for (int r = 0; r < repeat; r++) {
		checkCudaErrors(cudaMemcpyAsync(dest, src, sizeof(int)*num_elems, cudaMemcpyHostToDevice, streamToRun));
	}
}

void performDeviceToHostCopy(int *dest, int *src, int num_elems, int repeat, cudaStream_t streamToRun)
{
	for (int r = 0; r < repeat; r++) {
		checkCudaErrors(cudaMemcpyAsync(dest, src, sizeof(int)*num_elems, cudaMemcpyDeviceToHost, streamToRun));
	}
}

void performDeviceToDeviceCopy(int *dest, int *src, int num_elems, int repeat, cudaStream_t streamToRun, int deviceID)
{
	for (int r = 0; r < repeat; r++) {
		checkCudaErrors(cudaMemcpyAsync(dest, src, sizeof(int)*num_elems, cudaMemcpyDeviceToDevice, streamToRun));

	}
}


void outputLatencyMatrix(int numGPUs, bool p2p, memcpyKind kind)
{
    int repeat = MEMCOPY_ITERATIONS_For_Test_Latency;
    volatile int *flag = NULL;
    StopWatchInterface *stopWatch = NULL;
    std::vector<int *> buffersH(numGPUs);  // buffer for D2D, that is, intra-GPU copy
    std::vector<int *> buffersD(numGPUs);
    std::vector<int *> buffersD2D(numGPUs);
    std::vector<cudaStream_t> stream(numGPUs);
    std::vector<cudaEvent_t> start(numGPUs);
    std::vector<cudaEvent_t> stop(numGPUs);

    cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
    cudaCheckError();

    if (!sdkCreateTimer(&stopWatch)) {
        printf("Failed to create stop watch\n");
        exit(EXIT_FAILURE);
    }
    sdkStartTimer(&stopWatch);

    for (int d = 0; d < numGPUs; d++) {
        cudaSetDevice(d);
        cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
        cudaHostAlloc((void **)&buffersH[d], sizeof(int), cudaHostAllocPortable);
        cudaMalloc(&buffersD[d], sizeof(int));
        cudaMalloc(&buffersD2D[d], sizeof(int));
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    std::vector<double> gpuLatencyMatrix(numGPUs);
    std::vector<double> cpuLatencyMatrix(numGPUs);

    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(stream[i]);
        cudaCheckError();

        // Block the stream until all the work is queued up
        // DANGER! - cudaMemcpy*Async may infinitely block waiting for
        // room to push the operation, so keep the number of repeatitions
        // relatively low.  Higher repeatitions will cause the delay kernel
        // to timeout and lead to unstable results.
        *flag = 0;
        delay<<< 1, 1, 0, stream[i]>>>(flag);
        cudaCheckError();
        cudaEventRecord(start[i], stream[i]);

        sdkResetTimer(&stopWatch);

        if (kind == HOST_TO_DEVICE)
        {
            performHostToDeviceCopy(buffersD[i], buffersH[i], 1, repeat, stream[i]);
        }
        else if (kind == DEVICE_TO_HOST)
        {
            performDeviceToHostCopy(buffersH[i], buffersD[i], 1, repeat, stream[i]);
        }
        else if (kind == DEVICE_TO_DEVICE)
        {
        	performDeviceToDeviceCopy(buffersD2D[i], buffersD[i], 1, repeat, stream[i], i);
        }

        cudaEventRecord(stop[i], stream[i]);
        // Now that the work has been queued up, release the stream
        *flag = 1;
        cudaStreamSynchronize(stream[i]);
        cudaCheckError();

        float cpu_time_ms = sdkGetTimerValue(&stopWatch);

        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms, start[i], stop[i]);

        gpuLatencyMatrix[i] = (double)gpu_time_ms * 1e3 / repeat;
        cpuLatencyMatrix[i] = (double)cpu_time_ms * 1e3 / repeat;

    }

    printf("  GPU side latency\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%8d ", i);
    }
    printf("\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%8.05f ", gpuLatencyMatrix[i]);
    }
    printf("\n");

    printf("  CPU side latency\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%8d ", i);
    }
    printf("\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%8.05f ", cpuLatencyMatrix[i]);
    }
    printf("\n");

    for (int d = 0; d < numGPUs; d++) {
        cudaSetDevice(d);
        cudaCheckError();
        cudaFreeHost(buffersH[d]);
        cudaCheckError();
        cudaFree(buffersD[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
        cudaStreamDestroy(stream[d]);
        cudaCheckError();
    }

    sdkDeleteTimer(&stopWatch);

    cudaFreeHost((void *)flag);
    cudaCheckError();
}


///////////////////////////////////////////////////////////////////////////
//Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void)
{
    printf("Usage:  bandwidthTest [OPTION]...\n");
    printf("Test the bandwidth for device to host, host to device, and device to device transfers\n");
    printf("\n");
    printf("Example:  measure the bandwidth of device to host pinned memory copies in the range 1024 Bytes to 102400 Bytes in 1024 Byte increments\n");
    printf("./bandwidthTest --memory=pinned --mode=range --start=1024 --end=102400 --increment=1024 --dtoh\n");

    printf("\n");
    printf("Options:\n");
    printf("--help\tDisplay this help menu\n");
    printf("--csv\tPrint results as a CSV\n");
    printf("--device=[deviceno]\tSpecify the device device to be used\n");
    printf("  all - compute cumulative bandwidth on all the devices\n");
    printf("  0,1,2,...,n - Specify any particular device to be used\n");
    printf("--memory=[MEMMODE]\tSpecify which memory mode to use\n");
    printf("  pageable - pageable memory\n");
    printf("  pinned   - non-pageable system memory\n");
    printf("--mode=[MODE]\tSpecify the mode to use\n");
    printf("  quick - performs a quick measurement\n");
    printf("  range - measures a user-specified range of values\n");
    printf("  shmoo - performs an intense shmoo of a large range of values\n");

    printf("--htod\tMeasure host to device transfers\n");
    printf("--dtoh\tMeasure device to host transfers\n");
    printf("--dtod\tMeasure device to device transfers\n");
#if CUDART_VERSION >= 2020
    printf("--wc\tAllocate pinned memory as write-combined\n");
#endif
    printf("--cputiming\tForce CPU-based timing always\n");

    printf("Range mode options\n");
    printf("--start=[SIZE]\tStarting transfer size in bytes\n");
    printf("--end=[SIZE]\tEnding transfer size in bytes\n");
    printf("--increment=[SIZE]\tIncrement size in bytes\n");
}
