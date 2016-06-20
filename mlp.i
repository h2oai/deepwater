
%include "arrays_java.i"
%apply float[] {float *};
%apply float[] {mx_float *};
%apply int[] {int *};

%include "various.i"
%apply char **STRING_ARRAY { char ** }

%module mlp

%{
#include "mlp.hpp"
%}

%include "mlp.hpp"
