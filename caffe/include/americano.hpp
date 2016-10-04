#ifndef AMERICANO_HPP
#define AMERICANO_HPP

#include <boost/thread.hpp>
#include <csignal>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>

#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/db.hpp"
#include "google/protobuf/message.h"

namespace caffe {

using std::cout;
using std::string;
using google::protobuf::Message;
using google::protobuf::RepeatedPtrField;

static void sigstop() {
  raise(SIGSTOP);
}

static int device_count() {
  int count;
  CUDA_CHECK(::cudaGetDeviceCount(&count));
  return count;
}
static int get_device() {
  int device;
  CUDA_CHECK(::cudaGetDevice(&device));
  return device;
}
static void set_device(int device) {
  CUDA_CHECK(::cudaSetDevice(device));
}

static int ptr_device(float* ptr) {
  ::cudaPointerAttributes attributes;
  CUDA_CHECK(::cudaPointerGetAttributes(&attributes, ptr));
  return attributes.device;
}

class FloatNCCL;

class Barrier {
 public:
  Barrier(int count) :
    boost_(count) {
  }

  void Wait() {
    boost_.wait();
  }

 protected:
  boost::barrier boost_;

  friend class FloatNCCL;
};

class FloatNCCL {
 public:
  FloatNCCL(shared_ptr<Solver<float> > solver, Barrier* barrier)
    : caffe_(solver) {
    caffe_.set_barrier(&barrier->boost_);
    solver->add_callback(&caffe_);
    if (solver->param().layer_wise_reduce())
      solver->net()->add_after_backward(&caffe_);
  }

  static void InitSingleProcess(const vector<FloatNCCL*>& nccls) {
    vector<NCCL<float>*> caffes(nccls.size());
    for (int i = 0; i < nccls.size(); ++i) {
      caffes[i] = &nccls[i]->caffe_;
      LOG(INFO) << "nccl dev " << i << ": " << ptr_device(nccls[i]->caffe_.data());
    }
    NCCL<float>::InitSingleProcess(&caffes);
  }

  void Broadcast() {
    caffe_.Broadcast();
  }

 protected:
  NCCL<float> caffe_;
};

}

#endif
