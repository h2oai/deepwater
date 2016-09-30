#ifndef DEEPWATER_SWIG_TEST_HPP
#define DEEPWATER_SWIG_TEST_HPP

#include <string>
#include <vector>

void int_input_test(int i);

int int_return_test();

void int_arr_input_test(int * arr, int l);

void float_input_test(float i);

float float_return_test();

void float_arr_input_test(float * arr, int l);

std::vector<float> float_arr_return_test();

void string_input_test(char *);

std::string string_return_test();
#endif
