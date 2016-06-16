#ifndef __H2O_MPL_H__
#define __H2O_MPL_H__

#include <iostream>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;

class MLPClass {
 public:
  void train(float * aptr_x, float * aptr_y);
};

#endif  