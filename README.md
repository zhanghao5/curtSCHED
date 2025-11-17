# Intercept GPU cuda kernel calls and cu_sched scheduling framework

## compilation 
### make in the src code directory

## In the makefile ORIN flag means the cuda 12 support in ARC where the cuda API slightly changed compared to CUDA 11.4

## How to intercept your cuda code
  $ LD_PRELOAD=directoryto/cusched.so [your binary w/ cuda]
