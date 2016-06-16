#include <iostream>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;

class MLPClass {
 public:
  void train(float * aptr_x, float * aptr_y);
};
