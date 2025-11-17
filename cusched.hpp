#ifndef CUSCHED_HPP
#define CUSCHED_HPP
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//#include <cudaTypedefs.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <driver_types.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <queue>
#include <algorithm>
#include <semaphore.h> 

// project headers
#include "utils.hpp"
#include "model.hpp"
#include "timing.hpp"

typedef struct task_s{
    int task_id;
    // Parameters for cuLaunchKernel
    CUfunction f;
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void** kernelParams;
    void** extra;
    // Custom comparator (lower thread_id first)
    bool operator<(const task_s& other) const {
        return task_id > other.task_id;
    }
} task;

void cu_sched_init();
void cu_sched_finit();
void all_tasks_init();
void* defer_server(void *arg);

#endif
