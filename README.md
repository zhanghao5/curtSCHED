# Intercept GPU cuda kernel calls and cu_sched scheduling framework

## compilation 
### make in the src code directory
### make clean to remove .so file

## In the makefile ORIN flag means the cuda 12 support in ARC where the cuda API slightly changed compared to CUDA 11.4

## How to intercept your cuda code
  $ LD_PRELOAD=directoryto/cusched.so [your binary w/ cuda]

## sample config file
### filename cusched.config  mode 0 for model training; model 1 for online scheduling
mode=1  
period=66666666  
realtimecalls=141  
confidence=0.95  
