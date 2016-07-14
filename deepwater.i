
%include "arrays_java.i"
%apply float[] {float *};
%apply float[] {mx_float *};
%apply int[] {int *};

%include "various.i"
%apply char **STRING_ARRAY { char ** }

%typemap(jni) std::vector<float> "jfloatArray"
%typemap(jtype) std::vector<float> "float[]"
%typemap(jstype) std::vector<float> "float[]"
%typemap(in) std::vector<float> (std::vector<float> vecd) {
    if (!$input) {
        SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
        return $null;
    }
    const jsize sz = jenv->GetArrayLength($input);
    vecd.resize(sz);
    jfloat * const jarr = jenv->GetFloatArrayElements($input, 0);
    if (!jarr) return $null;
    for ( jsize i = 0; i < sz; i++ )
        vecd[i] = jarr[i];
    $1 = &vecd;
}

%typemap(out) std::vector<float> {
    const jsize sz = $1.size();
    $result = jenv->NewFloatArray(sz);
    jfloat* const jarr = jenv->GetFloatArrayElements($result, 0);
    if (!jarr) return $null;
    for ( jsize i = 0; i < sz; i++ )
        jarr[i] = $1[i];
    jenv->ReleaseFloatArrayElements($result, jarr, 0);
}
%typemap(javain) std::vector<float> "$javainput"
%typemap(javaout) std::vector<float> { return $jnicall; }

%apply std::vector<float> { std::vector<float> const & };

%module MLP

%{
#include "mlp.hpp"
%}

%include "mlp.hpp"


%module ImageTrain

%{
#include "image_train.hpp"
%}

%include "image_train.hpp"


%module ImagePred

%{
#include "image_pred.hpp"
%}

%include "image_pred.hpp"
