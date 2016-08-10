#include <iostream>
#include <cmath>
#include <stdlib.h>

#include "swigtest.hpp"

#define LG std::cout << "[" << __TIME__ << "] " \
    << __FILE__ << ":" << __LINE__ << ": "

void int_input_test(int i) {
  if (i == 42)
    LG << "int_input_test passed" << std::endl;
  else
    LG << "int_input_test failed" << std::endl;
}

int int_return_test() {
  return -35;
}

void int_arr_input_test(int * arr, int l) {
  std::vector<int> arr_cpp = {-35, 527, -799, 768, 940,
    -245, 11, -459, -258, 186
  };
  for (int i = 0; i < l; i++) {
    if(arr[i] != arr_cpp[i]) {
      LG << "int_arr_input_test failed" << std::endl;
      return;
    }
  }
  LG << "int_arr_input_test passed" << std::endl;
}

void float_input_test(float i) {
  if (std::fabs(i - 42.42) < 1e-5)
    LG << "float_input_test passed" << std::endl;
  else
    LG << "float_input_test failed" << std::endl;
}

float float_return_test() {
  return 42.42;
}

void float_arr_input_test(float * arr, int l) {
  std::vector<float> test = {0.903372, 0.451257, 0.368783,
    0.381643, 0.275748, 0.690426, 0.463654, 0.76209,
    0.782902, 0.998179
  };
  for (int i = 0; i < l; i++) {
    if (std::fabs(test[i] - arr[i]) > 1e-5) {
      LG << "float_arr_input_test failed" << std::endl; 
    }
  }
  LG << "float_arr_input_test passed" << std::endl;
} 

std::vector<float> float_arr_return_test() {
  srand(42);
  std::vector<float> res;
  int n = 10;
  for (int i = 0; i < n; i++) {
    res.push_back(rand() % 1000 / 1000.0);
  }
  return res;
}

void string_input_test(char * c) {
  std::string t = std::string(c);
  if (t == "string_input_test")
    LG << "string_input_test passed" << std::endl;
  else
    LG << "string_input_test failed" << std::endl; 
}

std::string string_return_test() {
  return std::string("string_return_test");
}

