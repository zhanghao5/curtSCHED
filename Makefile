CC := gcc
CXX := g++
CFLAGS := -g -fPIC -fpermissive -std=c++11 -O3 -I/usr/local/cuda-12.2/include
LDFLAGS := -shared -Wl,-rpath,/usr/local/cuda-12.2/lib64 -ldl -lpthread -lunwind

# Source files and object files
SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

# Debug flag
DEBUGF = 0 
ifeq ($(DEBUGF), 1)
CFLAGS += -DDEBUG
else
endif

# ORIN flag for CUDA 12 support
ORIN = 1
ifeq ($(ORIN), 1)
CFLAGS += -DORIN
endif

# Default target
all: cusched.so

# Rule to build the shared library
cusched.so: $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Rule to build object files
%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -rf *.o *.so

.PHONY: all clean

