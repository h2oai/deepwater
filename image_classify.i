
%include "arrays_java.i"
%apply float[] {float *};
%apply float[] {mx_float *};
%apply int[] {int *};

%include "various.i"
%apply char **STRING_ARRAY { char ** }

%module image_classify

%{
#include "image_classify.hpp"
%}

%include "image_classify.hpp"
