
%include "arrays_java.i"
%apply float[] {float *};

%module mlp

%{
#include "mlp.hpp"
%}

%include "mlp.hpp"
