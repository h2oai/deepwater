
%include "arrays_java.i"
%apply float[ANY] {float *};

%module mlp

%{
#include "mlp.hpp"
%}

%include "mlp.hpp"
