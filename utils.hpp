#ifndef UTILS_HPP
#define UTILS_HPP

#define __STDC_FORMAT_MACROS
#include<vector>
#include<iostream>
#include <inttypes.h>

#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <cxxabi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

using namespace std;

typedef struct {
    int mode;                   // 0 for train timing model, 1 for online schedule
    int n_calls;              // number of calls on various GPU architectures
    float confidence;           // model predict confidence score
    struct timespec period;     // schedule period
} cu_config;

template <typename T>
class RingBuffer {
public:
    RingBuffer(size_t capacity)
        : buffer_(capacity), capacity_(capacity), head_(0), tail_(0), size_(0) {}

    bool push(const T& item) {
        if (full()) return false;

        buffer_[tail_] = item;
        tail_ = (tail_ + 1) % capacity_;
        ++size_;
        return true;
    }

    bool pop() {
        if (empty()) return false;

        head_ = (head_ + 1) % capacity_;
        --size_;
        return true;
    }

    T front() const {
        if (empty()) throw std::runtime_error("Queue is empty!");
        return buffer_[head_];
    }

    bool empty() const {
        return size_ == 0;
    }

    bool full() const {
        return size_ == capacity_;
    }

    size_t size() const {
        return size_;
    }

private:
    std::vector<T> buffer_;
    size_t capacity_;
    size_t head_;
    size_t tail_;
    size_t size_;
};

int findInStack ( char *fname);

void load_config(const char* filename, cu_config* config);

#endif
