#include "cusched.hpp"

using namespace std;

// the number of threads that handle by cu_sched
#define EXPECTED_TASKS 4

volatile bool is_init = false;
bool is_set_init= false;
int active_tasks = 0;
static cu_config* config = NULL;
static int n_calls = 0;

//defer queue sync mutex and condition variable
static const int MAX_DEFER_QUEUE_SIZE = 64;
static pthread_mutex_t defer_m =  PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t defer_cond = PTHREAD_COND_INITIALIZER;
static RingBuffer<task> defer_queue(MAX_DEFER_QUEUE_SIZE);     // queue for deferred kernel launches, BE tasks


// tid set mutex and condition variable
static pthread_mutex_t set_m =  PTHREAD_MUTEX_INITIALIZER;    // mutex for tid set ready
static pthread_cond_t set_cond = PTHREAD_COND_INITIALIZER;    // condition variable for tid set ready
static vector<pid_t> set_tids;
static int n_threads = 0;
pthread_t thread_id = 0;                               // thread id for defer_server thread

//high low priority task sync mutex and condition variable
static pthread_mutex_t rt_m =  PTHREAD_MUTEX_INITIALIZER;   // mutex for sync between high and low priority tasks
static pthread_cond_t rt_cond = PTHREAD_COND_INITIALIZER;   // condition variable for sync between high and low priority tasks
static int turn_for_high = 1;  // high goes first
static int high_done = 0;
static int low_done = 0;
static int high_cnt = 0;
static int low_cnt = 0;
static int total_cnt = 0;    // total count to stop the experiment

// timinig mode
timing_info *ti = new timing_info();
struct timespec t_start = {0, 0};
struct timespec t_end= {0, 0};
struct timespec *now;
struct timespec *idle;
struct timespec *ptrNextRelease;

// measurement
static int miss_cnt = 0;
static int train_cnt = 0; 
static long idle_time = 0;
static long exec_time = 0;

sem_t sem;
sem_t sem_q;


//queue for deferred kernel launches, BE tasks

volatile CUresult (*real_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) = NULL;



extern "C" {
extern void *__libc_dlsym (void *, const char *);
}

void cu_sched_init() {
    if (is_init) return;
#ifdef DEBUG
    printf("init cuda kernel scheduler ... \n");
#endif
    config = (cu_config*)calloc(1, sizeof(cu_config));
    if (config == NULL) {
        printf("config data allocated failed \n");
        exit(1);
    }
    load_config("cusched.config", config);
    // Print the loaded values, mode = 0, train model, 1, online scheduler, 2, check num of calls on arcs
    cout << "Mode: " << config->mode << endl;
    cout << "Period: " << config->period.tv_nsec << " ns" << endl;
    cout << "Realtime Calls: " << config->n_calls << endl;
    cout << "Confidence: " << config->confidence << endl;
    pid_t task_id = syscall(__NR_gettid);
    if(config->mode == 1) {
        ptrNextRelease = (struct timespec *)malloc(sizeof(struct timespec));
        now = (struct timespec *)malloc(sizeof(struct timespec));
        idle = (struct timespec *)malloc(sizeof(struct timespec));
        ti->load_table("timing_table");
        pthread_create(&thread_id, NULL, defer_server, NULL);
        int ret = sem_init(&sem, 0, 0); 
        if(ret != 0){
            printf("semaphores init error!\n");
        }
        ret = sem_init(&sem_q, 0, 0);
        if (ret != 0){
            printf("semaphores init error!\n");
        }
    }

// #ifdef DEBUG
    printf("thread id : %d \n", task_id);
// #endif
    // register the fini function to free memory and dump out info
    const int result = std::atexit(cu_sched_finit);
    if (result != 0){
        std::cerr << "Registration failed\n";
        return EXIT_FAILURE;
    }
    if (config->mode == 1) {
        clock_gettime(CLOCK_REALTIME, now);
        plus_clock(ptrNextRelease, now, &(config->period));
    }

    is_init = true;
}

void cu_sched_finit() {
    // dump out infos
    if (config->mode == 0){
        ti->save_table("timing_table");
        ti->save_histogram("timing_histogram");
    }
    if (config->mode == 1) {
        free(ptrNextRelease);
        free(now);
        free(idle);
        printf("missed cnt : %d\n", miss_cnt);
        printf("train cnt : %d\n", train_cnt);
    }

    if (config->mode == 2){
        printf("number of culaunchkernel calls: %d \n", n_calls);
    }
    // wait for defer server complete
    // pthread_join(thread_id, NULL);

    // free allocated memories
    free(ti);
    free(config);
}

void* defer_server(void* arg) {
    const char* funcName = "NULL";
    while(1) {
        // pthread_mutex_lock(&defer_m);
        // pthread_cond_wait(&defer_cond, &defer_m);
        // printf("defer server start \n");
        sem_wait(&sem_q);

        while (!defer_queue.empty()) {
            task job = defer_queue.front();
            funcName = *(const char**)((uintptr_t)job.f + 8);
            string func(funcName);
            exec_time = ti->get_mean_time(func, job.gridDimX, job.blockDimX);
            // printf("predicted time: %ld \n", exec_time);
            clock_gettime(CLOCK_REALTIME, now);
            minus_clock(idle, ptrNextRelease, now);
            idle_time = idle->tv_nsec/1000 + idle->tv_sec * 1000000;
            // printf("idle_time: %ld \n", idle_time);
            if(exec_time > idle_time) {
                // pthread_cond_wait(&defer_cond, &defer_m);
                sem_wait(&sem_q);
                // printf("waiting for next schedule period\n");
            }
            else {
                // printf("launching kernel %s\n", func);
                // real_cuLaunchKernel(job.f, job.gridDimX, job.gridDimY, job.gridDimZ,
                //     job.blockDimX, job.blockDimY, job.blockDimZ,
                //     job.sharedMemBytes, job.hStream, job.kernelParams, job.extra);
                usleep(exec_time);
                train_cnt++;
                defer_queue.pop();
            }
        }
        // pthread_mutex_unlock(&defer_m);
    }
    return NULL;
}

void tid_set_init() {
    if(is_set_init) return;
    // If initialization completed while we were waiting for the lock

    pthread_mutex_lock(&set_m);
    pid_t cur_tid = syscall(__NR_gettid);
    set_tids.push_back(cur_tid);
    printf("tid %d is inserted\n", cur_tid);
    printf("current set size: %d , tid : %d\n", set_tids.size(), syscall(__NR_gettid));
    if (set_tids.size() == EXPECTED_TASKS) {
        // if all tasks are ready, signal the condition variable
        is_set_init = true;
        sort(set_tids.begin(), set_tids.end());
        printf("sorted tids: %d, %d, %d, %d\n", set_tids[0], set_tids[1], set_tids[2], set_tids[3]);
        pthread_cond_broadcast(&set_cond);
        printf("all tasks are ready\n");
    }
    else {
        while (!is_set_init) {
            printf("waiting for other tasks to be ready, current tid: %d\n", syscall(__NR_gettid));
            pthread_cond_wait(&set_cond, &set_m);
        }
        printf("waking up tid %d\n", cur_tid);
    }
    pthread_mutex_unlock(&set_m);
}

extern "C" CUresult
// CUresult
cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY,
                unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY,
                unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream,
                void** kernelParams, void** extra ){
    CUresult ret;
#ifdef DEBUG
    printf("============================= intercepting cuLaunchKernel =================================\n");
    printf("kernel function %x, xyz: %u, %u, %u, bxyz : %u, %u, %u\n", f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);
#endif
    // declare CUresult
    if (!is_init) cu_sched_init();
    if (config->mode == 1 || !is_set_init) tid_set_init();
    CUresult (*real_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) = dlsym(RTLD_NEXT, "cuLaunchKernel");

    const char* funcName = *(const char**)((uintptr_t)f + 8);
    string func(funcName);
#ifdef DEBUG
    printf("calling kernel func %x via launch %x\n", f, real_cuLaunchKernel);
#endif
    // Get thread ID as priority
    pid_t pid = syscall(__NR_gettid);
    // printf("before entering the scheduler the pid : %d\n", pid);


    if (config->mode == 0) {   // train mode
        // train mode
        clock_gettime(CLOCK_REALTIME, &t_start);
        ret = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &t_end);
        struct timespec tmp = {0,0};
        minus_clock(&tmp, &t_end, &t_start);
#ifdef DEBUG
        printf("timing recording  sec nsec    %ld : %ld\n", tmp.tv_sec, tmp.tv_nsec);
        printf("func signature : %x\n", func);
#endif
        ti->record(func, gridDimX, blockDimX, tmp.tv_nsec/1000);    // in mircrosec

    }
    else if (config->mode == 1) {  // online scheduler
      // defer_queue data structure include the func in string data type
      if(set_tids.size() != EXPECTED_TASKS) {
        printf("test set_tids size: %d not equals to the expected value!\n", set_tids.size());
        exit(1);
      }
      if (pid == set_tids[0]) {       // high priority RT task
        // pthread_mutex_lock(&rt_m);
        while(!turn_for_high) {
            // pthread_cond_wait(&rt_cond, &rt_m);
            sem_wait(&sem);
        }
        ret = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
        high_cnt++;
        total_cnt++;
        if(high_cnt == config->n_calls) {
            turn_for_high = 0;
            high_done = 1;
            high_cnt = 0;
            // pthread_cond_broadcast(&rt_cond);
            sem_post(&sem);
        }
        // pthread_mutex_unlock(&rt_m);
        // clock_gettime(CLOCK_REALTIME, now);
        // minus_clock(idle, ptrNextRelease, now);
        // clock_nanosleep(CLOCK_REALTIME, 0, idle, 0);
        
      } // low priority RT task
      else if (pid == set_tids[1]) {   //  low priority RT task
        // pthread_mutex_lock(&rt_m);
        while(turn_for_high) {
            // pthread_cond_wait(&rt_cond, &rt_m);
            sem_wait(&sem);
        }
        ret = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
        low_cnt++;
        if(low_cnt == config->n_calls) {
            low_done = 1;
            low_cnt = 0;
            turn_for_high = 1;
            // pthread_cond_broadcast(&rt_cond);
            sem_post(&sem);       
        }
        // pthread_mutex_unlock(&rt_m);
        if (total_cnt == 1000 * config->n_calls) {
            printf("RT tasks are donw, stop experiments\n");
            exit(1);
        }
        

      } 
      else {                          // BE tasks
        // pthread_mutex_lock(&defer_m);
        while(defer_queue.full()) {
            // block the call
            // pthread_cond_wait(&defer_cond, &defer_m);
            sem_wait(&sem_q);
        }
        defer_queue.push({pid, f, gridDimX, gridDimY, gridDimZ,
                       blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                       hStream, kernelParams, extra});

        
        // pthread_mutex_unlock(&defer_m);
        return CUDA_SUCCESS;   // since these are cascaed culaunchkernel calls, we can return success
      }

      // All RT tasks are done
      if (high_done && low_done) {
        clock_gettime(CLOCK_REALTIME, now);
        minus_clock(idle, ptrNextRelease, now);
        idle_time = idle->tv_nsec/1000 + idle->tv_sec * 1000000;
        if(idle_time <=0) {
            miss_cnt++;
            idle_time = 0;
#ifdef DEBUG
            printf("a misssed deadline for RT task\n");
#endif
        }
        high_done = 0;
        low_done = 0;
        // pthread_mutex_lock(&defer_m);
        // pthread_cond_broadcast(&defer_cond);
        // pthread_mutex_unlock(&defer_m);
        sem_post(&sem_q);
        clock_nanosleep(CLOCK_REALTIME, 0, idle, 0);
      }
      if(high_done == 0 && low_done == 0) {
        clock_gettime(CLOCK_REALTIME, now);
        plus_clock(ptrNextRelease, now, &(config->period));
#ifdef DEBUG      
        printf("update next deadline\n");
#endif
      }

    } else if (config->mode == 2) {   // architecture-aware profiling on per image n_calls
        ret = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
        n_calls++;
    
    }
    // printf("high_cnt: %d, low_cnt: %d\n", high_cnt, low_cnt);
    return ret;
}

// orin CUDA 11.4 =============== ARC Cuda 12     =============================
#ifdef ORIN

volatile CUresult (*real_cuGetProcAddress_v2) (const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) = NULL; // arc 12.2
CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) // new arc 12.2
{
    //CUresult (*real_cuGetProcAddress) (const char*, void**, int, cuuint64_t) = dlsym(RTLD_NEXT, "cuGetProcAddress");
#ifdef DEBUG1
    printf("looking up func %s via cuGetProcAddress_v2 %lx\n", symbol, (unsigned long int) cuGetProcAddress_v2);
    printf("real_cuGetProcAddress_v2 %lx\n", (unsigned long int) real_cuGetProcAddress_v2);
#endif
    while (!real_cuGetProcAddress_v2); // wait for init
    CUresult result = real_cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus); // arc 12.2
    void *handle;
    if(strcmp(symbol, "cuLaunchKernel") == 0) {
        if (real_cuLaunchKernel == NULL) {
            printf("================= get cuLaunchKernel =================\n");
            //void *dlhandle = dlopen("/lib/aarch64-linux-gnu/libdl.so.2", RTLD_LAZY);  // orin agx
            void *dlhandle = dlopen("/lib64/libdl.so.2", RTLD_LAZY);  // new arc
            void *(*real_dlsym) (void *, const char *) = __libc_dlsym(dlhandle, "dlsym");
            //handle = dlopen("/usr/lib/aarch64-linux-gnu/tegra/libcuda.so", RTLD_LAZY); // orin agx
            handle = dlopen("/lib64/libcuda.so", RTLD_LAZY);    // new arc
            real_cuLaunchKernel = real_dlsym(handle, symbol);
        }
        //    if (handle != RTLD_NEXT)
        *pfn = cuLaunchKernel;
#ifdef DEBUG1
        printf("real_cuGetProcAddress2 %lx\n", (unsigned long int) *pfn);
#endif
    }
    else if (strcmp(symbol, "cuGetProcAddress") == 0) {          // because see define cuGetProcAddress cu..._v2, try
    //else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {    // change from orin 11.4 to _v2
        //    if (handle != RTLD_NEXT)
        printf("aaaaah\n");
        *pfn = cuGetProcAddress_v2;                          // align change with upper comment
    }

#ifdef DEBUG1
    printf("returning func %lx for symbol %s\n", (unsigned long int) *pfn, symbol);
    printf("result %lx\n", (unsigned long int) result);
#endif
    return result;
}

extern void *__libc_dlsym (void *, const char *);
void *dlsym(void *handle, const char *symbol)
{
#ifdef DEBUG1
    printf("trapped into dlsym cuda 11.4 orin version\n");
#endif
    //void *dlhandle = dlopen("/lib/aarch64-linux-gnu/libdl.so.2", RTLD_LAZY);  // orin agax
    void *dlhandle = dlopen("/lib64/libdl.so.2", RTLD_LAZY);  // new arc
    void *(*real_dlsym) (void *, const char *) = __libc_dlsym(dlhandle, "dlsym"); /* now, this will call dlsym() library function */
#ifdef DEBUG1
    printf("real_dlsym %lx\n", (unsigned long int) real_dlsym);
#endif

    if (strcmp(symbol, "cuLaunchKernel") == 0) {
        if (real_cuLaunchKernel == NULL && handle != RTLD_NEXT)
            real_cuLaunchKernel = real_dlsym(handle, symbol);
        if (handle != RTLD_NEXT) {
            return cuLaunchKernel;
        }
        else {
#ifdef DEBUG1
            printf("result1 %lx and real %lx\n", (unsigned long int) cuLaunchKernel, (unsigned long int) real_cuLaunchKernel);
#endif
            return (void *)real_cuLaunchKernel;
        }
    }
    else if (strcmp(symbol, "cuGetProcAddress_v2") == 0 ) {
        if (real_cuGetProcAddress_v2 == NULL && handle != RTLD_NEXT)
        real_cuGetProcAddress_v2 = real_dlsym(handle, symbol);
        if (handle != RTLD_NEXT) {
#ifdef DEBUG
            printf("result2 %lx and real %lx\n", (unsigned long int) cuGetProcAddress_v2, (unsigned long int) real_cuGetProcAddress_v2);
#endif
            return cuGetProcAddress_v2;    // change from non to _v2
        }
        else {
#ifdef DEBUG
            printf("result3 %lx\n", (unsigned long int) real_cuGetProcAddress_v2);
#endif
            return (void *)real_cuGetProcAddress_v2;
        }
    }
#ifdef DEBUG
    printf("mydlsym searching for handle %lx symbol %s\n", (unsigned long int) handle, symbol);
#endif
    void* result = real_dlsym(handle, symbol);
#ifdef DEBUG
    printf("result %lx\n", (unsigned long int) result);
#endif
    return result;
}

/* =====================  end of cuda 12 ======================================*/
#else

extern "C" {
extern void *__libc_dlsym (void *, const char *);
void *dlsym(void *handle, const char *symbol)
{
    printf("aaah! come in\n");
    // new for test
    //void *dlhandle = dlopen("/usr/lib64/libdl.so", RTLD_LAZY);   //old arc
    void *dlhandle = dlopen("/lib64/libdl-2.28.so", RTLD_LAZY);    // new arc
    void *(*real_dlsym) (void *, const char *) = __libc_dlsym(dlhandle, "dlsym");
    if (strcmp(symbol, "cudaMemcpyAsync") == 0) {
        if (real_cudaMemcpyAsync == NULL && handle != RTLD_NEXT)
            real_cudaMemcpyAsync = real_dlsym(handle, symbol);
        if (handle != RTLD_NEXT) {
            return cudaMemcpyAsync;
        }
    }
    else if (strcmp(symbol, "cuLaunchKernel") == 0) {
        if (real_cuLaunchKernel == NULL && handle != RTLD_NEXT)
            real_cuLaunchKernel = real_dlsym(handle, symbol);
        if (handle != RTLD_NEXT) {
            return cuLaunchKernel;
        }
        else {
            return (void *)real_cuLaunchKernel;
        }
    }
    void* result = real_dlsym(handle, symbol);
    return result;

#endif




