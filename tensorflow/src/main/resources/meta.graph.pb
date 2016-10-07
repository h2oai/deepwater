meta_info_def {
  stripped_op_list {
    op {
      name: "Add"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      input_arg {
        name: "y"
        type_attr: "T"
      }
      output_arg {
        name: "z"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_UINT8
            type: DT_INT8
            type: DT_INT16
            type: DT_INT32
            type: DT_INT64
            type: DT_COMPLEX64
            type: DT_COMPLEX128
            type: DT_STRING
          }
        }
      }
    }
    op {
      name: "ApplyAdagrad"
      input_arg {
        name: "var"
        type_attr: "T"
        is_ref: true
      }
      input_arg {
        name: "accum"
        type_attr: "T"
        is_ref: true
      }
      input_arg {
        name: "lr"
        type_attr: "T"
      }
      input_arg {
        name: "grad"
        type_attr: "T"
      }
      output_arg {
        name: "out"
        type_attr: "T"
        is_ref: true
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_INT64
            type: DT_INT32
            type: DT_UINT8
            type: DT_UINT16
            type: DT_INT16
            type: DT_INT8
            type: DT_COMPLEX64
            type: DT_COMPLEX128
            type: DT_QINT8
            type: DT_QUINT8
            type: DT_QINT32
            type: DT_HALF
          }
        }
      }
      attr {
        name: "use_locking"
        type: "bool"
        default_value {
          b: false
        }
      }
    }
    op {
      name: "Assign"
      input_arg {
        name: "ref"
        type_attr: "T"
        is_ref: true
      }
      input_arg {
        name: "value"
        type_attr: "T"
      }
      output_arg {
        name: "output_ref"
        type_attr: "T"
        is_ref: true
      }
      attr {
        name: "T"
        type: "type"
      }
      attr {
        name: "validate_shape"
        type: "bool"
        default_value {
          b: true
        }
      }
      attr {
        name: "use_locking"
        type: "bool"
        default_value {
          b: true
        }
      }
      allows_uninitialized_input: true
    }
    op {
      name: "BiasAdd"
      input_arg {
        name: "value"
        type_attr: "T"
      }
      input_arg {
        name: "bias"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_INT64
            type: DT_INT32
            type: DT_UINT8
            type: DT_UINT16
            type: DT_INT16
            type: DT_INT8
            type: DT_COMPLEX64
            type: DT_COMPLEX128
            type: DT_QINT8
            type: DT_QUINT8
            type: DT_QINT32
            type: DT_HALF
          }
        }
      }
      attr {
        name: "data_format"
        type: "string"
        default_value {
          s: "NHWC"
        }
        allowed_values {
          list {
            s: "NHWC"
            s: "NCHW"
          }
        }
      }
    }
    op {
      name: "BiasAddGrad"
      input_arg {
        name: "out_backprop"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_INT64
            type: DT_INT32
            type: DT_UINT8
            type: DT_UINT16
            type: DT_INT16
            type: DT_INT8
            type: DT_COMPLEX64
            type: DT_COMPLEX128
            type: DT_QINT8
            type: DT_QUINT8
            type: DT_QINT32
            type: DT_HALF
          }
        }
      }
      attr {
        name: "data_format"
        type: "string"
        default_value {
          s: "NHWC"
        }
        allowed_values {
          list {
            s: "NHWC"
            s: "NCHW"
          }
        }
      }
    }
    op {
      name: "Concat"
      input_arg {
        name: "concat_dim"
        type: DT_INT32
      }
      input_arg {
        name: "values"
        type_attr: "T"
        number_attr: "N"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "N"
        type: "int"
        has_minimum: true
        minimum: 2
      }
      attr {
        name: "T"
        type: "type"
      }
    }
    op {
      name: "Const"
      output_arg {
        name: "output"
        type_attr: "dtype"
      }
      attr {
        name: "value"
        type: "tensor"
      }
      attr {
        name: "dtype"
        type: "type"
      }
    }
    op {
      name: "Conv2D"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      input_arg {
        name: "filter"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
          }
        }
      }
      attr {
        name: "strides"
        type: "list(int)"
      }
      attr {
        name: "use_cudnn_on_gpu"
        type: "bool"
        default_value {
          b: true
        }
      }
      attr {
        name: "padding"
        type: "string"
        allowed_values {
          list {
            s: "SAME"
            s: "VALID"
          }
        }
      }
      attr {
        name: "data_format"
        type: "string"
        default_value {
          s: "NHWC"
        }
        allowed_values {
          list {
            s: "NHWC"
            s: "NCHW"
          }
        }
      }
    }
    op {
      name: "Conv2DBackpropFilter"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      input_arg {
        name: "filter_sizes"
        type: DT_INT32
      }
      input_arg {
        name: "out_backprop"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
          }
        }
      }
      attr {
        name: "strides"
        type: "list(int)"
      }
      attr {
        name: "use_cudnn_on_gpu"
        type: "bool"
        default_value {
          b: true
        }
      }
      attr {
        name: "padding"
        type: "string"
        allowed_values {
          list {
            s: "SAME"
            s: "VALID"
          }
        }
      }
      attr {
        name: "data_format"
        type: "string"
        default_value {
          s: "NHWC"
        }
        allowed_values {
          list {
            s: "NHWC"
            s: "NCHW"
          }
        }
      }
    }
    op {
      name: "Conv2DBackpropInput"
      input_arg {
        name: "input_sizes"
        type: DT_INT32
      }
      input_arg {
        name: "filter"
        type_attr: "T"
      }
      input_arg {
        name: "out_backprop"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
          }
        }
      }
      attr {
        name: "strides"
        type: "list(int)"
      }
      attr {
        name: "use_cudnn_on_gpu"
        type: "bool"
        default_value {
          b: true
        }
      }
      attr {
        name: "padding"
        type: "string"
        allowed_values {
          list {
            s: "SAME"
            s: "VALID"
          }
        }
      }
      attr {
        name: "data_format"
        type: "string"
        default_value {
          s: "NHWC"
        }
        allowed_values {
          list {
            s: "NHWC"
            s: "NCHW"
          }
        }
      }
    }
    op {
      name: "ExpandDims"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      input_arg {
        name: "dim"
        type_attr: "Tdim"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
      }
      attr {
        name: "Tdim"
        type: "type"
        default_value {
          type: DT_INT32
        }
        allowed_values {
          list {
            type: DT_INT32
            type: DT_INT64
          }
        }
      }
    }
    op {
      name: "Fill"
      input_arg {
        name: "dims"
        type: DT_INT32
      }
      input_arg {
        name: "value"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
      }
    }
    op {
      name: "Identity"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
      }
    }
    op {
      name: "MatMul"
      input_arg {
        name: "a"
        type_attr: "T"
      }
      input_arg {
        name: "b"
        type_attr: "T"
      }
      output_arg {
        name: "product"
        type_attr: "T"
      }
      attr {
        name: "transpose_a"
        type: "bool"
        default_value {
          b: false
        }
      }
      attr {
        name: "transpose_b"
        type: "bool"
        default_value {
          b: false
        }
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_INT32
            type: DT_COMPLEX64
            type: DT_COMPLEX128
          }
        }
      }
    }
    op {
      name: "MaxPool"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        default_value {
          type: DT_FLOAT
        }
        allowed_values {
          list {
            type: DT_FLOAT
            type: DT_HALF
          }
        }
      }
      attr {
        name: "ksize"
        type: "list(int)"
        has_minimum: true
        minimum: 4
      }
      attr {
        name: "strides"
        type: "list(int)"
        has_minimum: true
        minimum: 4
      }
      attr {
        name: "padding"
        type: "string"
        allowed_values {
          list {
            s: "SAME"
            s: "VALID"
          }
        }
      }
      attr {
        name: "data_format"
        type: "string"
        default_value {
          s: "NHWC"
        }
        allowed_values {
          list {
            s: "NHWC"
            s: "NCHW"
          }
        }
      }
    }
    op {
      name: "MaxPoolGrad"
      input_arg {
        name: "orig_input"
        type_attr: "T"
      }
      input_arg {
        name: "orig_output"
        type_attr: "T"
      }
      input_arg {
        name: "grad"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "ksize"
        type: "list(int)"
        has_minimum: true
        minimum: 4
      }
      attr {
        name: "strides"
        type: "list(int)"
        has_minimum: true
        minimum: 4
      }
      attr {
        name: "padding"
        type: "string"
        allowed_values {
          list {
            s: "SAME"
            s: "VALID"
          }
        }
      }
      attr {
        name: "data_format"
        type: "string"
        default_value {
          s: "NHWC"
        }
        allowed_values {
          list {
            s: "NHWC"
            s: "NCHW"
          }
        }
      }
      attr {
        name: "T"
        type: "type"
        default_value {
          type: DT_FLOAT
        }
        allowed_values {
          list {
            type: DT_FLOAT
            type: DT_HALF
          }
        }
      }
    }
    op {
      name: "MergeSummary"
      input_arg {
        name: "inputs"
        type: DT_STRING
        number_attr: "N"
      }
      output_arg {
        name: "summary"
        type: DT_STRING
      }
      attr {
        name: "N"
        type: "int"
        has_minimum: true
        minimum: 1
      }
    }
    op {
      name: "Mul"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      input_arg {
        name: "y"
        type_attr: "T"
      }
      output_arg {
        name: "z"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_UINT8
            type: DT_INT8
            type: DT_UINT16
            type: DT_INT16
            type: DT_INT32
            type: DT_INT64
            type: DT_COMPLEX64
            type: DT_COMPLEX128
          }
        }
      }
      is_commutative: true
    }
    op {
      name: "NoOp"
    }
    op {
      name: "Pack"
      input_arg {
        name: "values"
        type_attr: "T"
        number_attr: "N"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "N"
        type: "int"
        has_minimum: true
        minimum: 1
      }
      attr {
        name: "T"
        type: "type"
      }
      attr {
        name: "axis"
        type: "int"
        default_value {
          i: 0
        }
      }
    }
    op {
      name: "Placeholder"
      output_arg {
        name: "output"
        type_attr: "dtype"
      }
      attr {
        name: "dtype"
        type: "type"
      }
      attr {
        name: "shape"
        type: "shape"
        default_value {
          shape {
          }
        }
      }
    }
    op {
      name: "RandomUniform"
      input_arg {
        name: "shape"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "dtype"
      }
      attr {
        name: "seed"
        type: "int"
        default_value {
          i: 0
        }
      }
      attr {
        name: "seed2"
        type: "int"
        default_value {
          i: 0
        }
      }
      attr {
        name: "dtype"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
          }
        }
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_INT32
            type: DT_INT64
          }
        }
      }
      is_stateful: true
    }
    op {
      name: "Reshape"
      input_arg {
        name: "tensor"
        type_attr: "T"
      }
      input_arg {
        name: "shape"
        type_attr: "Tshape"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
      }
      attr {
        name: "Tshape"
        type: "type"
        default_value {
          type: DT_INT32
        }
        allowed_values {
          list {
            type: DT_INT32
            type: DT_INT64
          }
        }
      }
    }
    op {
      name: "RestoreV2"
      input_arg {
        name: "prefix"
        type: DT_STRING
      }
      input_arg {
        name: "tensor_names"
        type: DT_STRING
      }
      input_arg {
        name: "shape_and_slices"
        type: DT_STRING
      }
      output_arg {
        name: "tensors"
        type_list_attr: "dtypes"
      }
      attr {
        name: "dtypes"
        type: "list(type)"
        has_minimum: true
        minimum: 1
      }
    }
    op {
      name: "SaveSlices"
      input_arg {
        name: "filename"
        type: DT_STRING
      }
      input_arg {
        name: "tensor_names"
        type: DT_STRING
      }
      input_arg {
        name: "shapes_and_slices"
        type: DT_STRING
      }
      input_arg {
        name: "data"
        type_list_attr: "T"
      }
      attr {
        name: "T"
        type: "list(type)"
        has_minimum: true
        minimum: 1
      }
    }
    op {
      name: "ScalarSummary"
      input_arg {
        name: "tags"
        type: DT_STRING
      }
      input_arg {
        name: "values"
        type_attr: "T"
      }
      output_arg {
        name: "summary"
        type: DT_STRING
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_INT32
            type: DT_INT64
            type: DT_UINT8
            type: DT_INT16
            type: DT_INT8
            type: DT_UINT16
            type: DT_HALF
          }
        }
      }
    }
    op {
      name: "Shape"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "out_type"
      }
      attr {
        name: "T"
        type: "type"
      }
      attr {
        name: "out_type"
        type: "type"
        default_value {
          type: DT_INT32
        }
        allowed_values {
          list {
            type: DT_INT32
            type: DT_INT64
          }
        }
      }
    }
    op {
      name: "Slice"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      input_arg {
        name: "begin"
        type_attr: "Index"
      }
      input_arg {
        name: "size"
        type_attr: "Index"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
      }
      attr {
        name: "Index"
        type: "type"
        allowed_values {
          list {
            type: DT_INT32
            type: DT_INT64
          }
        }
      }
    }
    op {
      name: "Softmax"
      input_arg {
        name: "logits"
        type_attr: "T"
      }
      output_arg {
        name: "softmax"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
          }
        }
      }
    }
    op {
      name: "SoftmaxCrossEntropyWithLogits"
      input_arg {
        name: "features"
        type_attr: "T"
      }
      input_arg {
        name: "labels"
        type_attr: "T"
      }
      output_arg {
        name: "loss"
        type_attr: "T"
      }
      output_arg {
        name: "backprop"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
          }
        }
      }
    }
    op {
      name: "Sub"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      input_arg {
        name: "y"
        type_attr: "T"
      }
      output_arg {
        name: "z"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_INT32
            type: DT_INT64
            type: DT_COMPLEX64
            type: DT_COMPLEX128
          }
        }
      }
    }
    op {
      name: "Tanh"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      output_arg {
        name: "y"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_COMPLEX64
            type: DT_COMPLEX128
          }
        }
      }
    }
    op {
      name: "TanhGrad"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      input_arg {
        name: "y"
        type_attr: "T"
      }
      output_arg {
        name: "z"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_COMPLEX64
            type: DT_COMPLEX128
          }
        }
      }
    }
    op {
      name: "Unpack"
      input_arg {
        name: "value"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
        number_attr: "num"
      }
      attr {
        name: "num"
        type: "int"
        has_minimum: true
      }
      attr {
        name: "T"
        type: "type"
      }
      attr {
        name: "axis"
        type: "int"
        default_value {
          i: 0
        }
      }
    }
    op {
      name: "Variable"
      output_arg {
        name: "ref"
        type_attr: "dtype"
        is_ref: true
      }
      attr {
        name: "shape"
        type: "shape"
      }
      attr {
        name: "dtype"
        type: "type"
      }
      attr {
        name: "container"
        type: "string"
        default_value {
          s: ""
        }
      }
      attr {
        name: "shared_name"
        type: "string"
        default_value {
          s: ""
        }
      }
      is_stateful: true
    }
    op {
      name: "ZerosLike"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      output_arg {
        name: "y"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
      }
    }
  }
}
graph_def {
  node {
    name: "input/Placeholder"
    op: "Placeholder"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 150528
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
        }
      }
    }
  }
  node {
    name: "input/Reshape/shape"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 4
            }
          }
          tensor_content: "\377\377\377\377\340\000\000\000\340\000\000\000\003\000\000\000"
        }
      }
    }
  }
  node {
    name: "input/Reshape"
    op: "Reshape"
    input: "input/Placeholder"
    input: "input/Reshape/shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 3
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights"
    op: "Variable"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 3
          }
          dim {
            size: 20
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights/Initializer/random_uniform/shape"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 4
            }
          }
          tensor_content: "\005\000\000\000\005\000\000\000\003\000\000\000\024\000\000\000"
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights/Initializer/random_uniform/min"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: -0.102150782943
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights/Initializer/random_uniform/max"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.102150782943
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights/Initializer/random_uniform/RandomUniform"
    op: "RandomUniform"
    input: "inference/conv1/Conv/weights/Initializer/random_uniform/shape"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "seed"
      value {
        i: 0
      }
    }
    attr {
      key: "seed2"
      value {
        i: 0
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights/Initializer/random_uniform/sub"
    op: "Sub"
    input: "inference/conv1/Conv/weights/Initializer/random_uniform/max"
    input: "inference/conv1/Conv/weights/Initializer/random_uniform/min"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights/Initializer/random_uniform/mul"
    op: "Mul"
    input: "inference/conv1/Conv/weights/Initializer/random_uniform/RandomUniform"
    input: "inference/conv1/Conv/weights/Initializer/random_uniform/sub"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights/Initializer/random_uniform"
    op: "Add"
    input: "inference/conv1/Conv/weights/Initializer/random_uniform/mul"
    input: "inference/conv1/Conv/weights/Initializer/random_uniform/min"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights/Assign"
    op: "Assign"
    input: "inference/conv1/Conv/weights"
    input: "inference/conv1/Conv/weights/Initializer/random_uniform"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/conv1/Conv/weights/read"
    op: "Identity"
    input: "inference/conv1/Conv/weights"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/Conv2D"
    op: "Conv2D"
    input: "input/Reshape"
    input: "inference/conv1/Conv/weights/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/conv1/Conv/biases"
    op: "Variable"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 20
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "inference/conv1/Conv/biases/Initializer/zeros"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 20
            }
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/biases/Assign"
    op: "Assign"
    input: "inference/conv1/Conv/biases"
    input: "inference/conv1/Conv/biases/Initializer/zeros"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/conv1/Conv/biases/read"
    op: "Identity"
    input: "inference/conv1/Conv/biases"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv1/Conv/BiasAdd"
    op: "BiasAdd"
    input: "inference/conv1/Conv/Conv2D"
    input: "inference/conv1/Conv/biases/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "inference/conv1/Conv/Tanh"
    op: "Tanh"
    input: "inference/conv1/Conv/BiasAdd"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv1/MaxPool2D/MaxPool"
    op: "MaxPool"
    input: "inference/conv1/Conv/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "ksize"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
    attr {
      key: "padding"
      value {
        s: "VALID"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights"
    op: "Variable"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 20
          }
          dim {
            size: 50
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights/Initializer/random_uniform/shape"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 4
            }
          }
          tensor_content: "\005\000\000\000\005\000\000\000\024\000\000\0002\000\000\000"
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights/Initializer/random_uniform/min"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: -0.0585540048778
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights/Initializer/random_uniform/max"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.0585540048778
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights/Initializer/random_uniform/RandomUniform"
    op: "RandomUniform"
    input: "inference/conv2/Conv/weights/Initializer/random_uniform/shape"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "seed"
      value {
        i: 0
      }
    }
    attr {
      key: "seed2"
      value {
        i: 0
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights/Initializer/random_uniform/sub"
    op: "Sub"
    input: "inference/conv2/Conv/weights/Initializer/random_uniform/max"
    input: "inference/conv2/Conv/weights/Initializer/random_uniform/min"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights/Initializer/random_uniform/mul"
    op: "Mul"
    input: "inference/conv2/Conv/weights/Initializer/random_uniform/RandomUniform"
    input: "inference/conv2/Conv/weights/Initializer/random_uniform/sub"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights/Initializer/random_uniform"
    op: "Add"
    input: "inference/conv2/Conv/weights/Initializer/random_uniform/mul"
    input: "inference/conv2/Conv/weights/Initializer/random_uniform/min"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights/Assign"
    op: "Assign"
    input: "inference/conv2/Conv/weights"
    input: "inference/conv2/Conv/weights/Initializer/random_uniform"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/conv2/Conv/weights/read"
    op: "Identity"
    input: "inference/conv2/Conv/weights"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/Conv2D"
    op: "Conv2D"
    input: "inference/conv1/MaxPool2D/MaxPool"
    input: "inference/conv2/Conv/weights/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/conv2/Conv/biases"
    op: "Variable"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 50
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "inference/conv2/Conv/biases/Initializer/zeros"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 50
            }
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/biases/Assign"
    op: "Assign"
    input: "inference/conv2/Conv/biases"
    input: "inference/conv2/Conv/biases/Initializer/zeros"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/conv2/Conv/biases/read"
    op: "Identity"
    input: "inference/conv2/Conv/biases"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv2/Conv/BiasAdd"
    op: "BiasAdd"
    input: "inference/conv2/Conv/Conv2D"
    input: "inference/conv2/Conv/biases/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "inference/conv2/Conv/Tanh"
    op: "Tanh"
    input: "inference/conv2/Conv/BiasAdd"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/conv2/MaxPool2D/MaxPool"
    op: "MaxPool"
    input: "inference/conv2/Conv/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 56
            }
            dim {
              size: 56
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "ksize"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
    attr {
      key: "padding"
      value {
        s: "VALID"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
  }
  node {
    name: "inference/flatten/Reshape/shape"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 2
            }
          }
          tensor_content: "\377\377\377\377\200d\002\000"
        }
      }
    }
  }
  node {
    name: "inference/flatten/Reshape"
    op: "Reshape"
    input: "inference/conv2/MaxPool2D/MaxPool"
    input: "inference/flatten/Reshape/shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 156800
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc1/Shape"
    op: "Shape"
    input: "inference/flatten/Reshape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "inference/fc1/unpack"
    op: "Unpack"
    input: "inference/fc1/Shape"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
          shape {
          }
        }
      }
    }
    attr {
      key: "axis"
      value {
        i: 0
      }
    }
    attr {
      key: "num"
      value {
        i: 2
      }
    }
  }
  node {
    name: "inference/fc1/weights"
    op: "Variable"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 156800
          }
          dim {
            size: 500
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "inference/fc1/weights/Initializer/random_uniform/shape"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 2
            }
          }
          tensor_content: "\200d\002\000\364\001\000\000"
        }
      }
    }
  }
  node {
    name: "inference/fc1/weights/Initializer/random_uniform/min"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: -0.00617605634034
        }
      }
    }
  }
  node {
    name: "inference/fc1/weights/Initializer/random_uniform/max"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.00617605634034
        }
      }
    }
  }
  node {
    name: "inference/fc1/weights/Initializer/random_uniform/RandomUniform"
    op: "RandomUniform"
    input: "inference/fc1/weights/Initializer/random_uniform/shape"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "seed"
      value {
        i: 0
      }
    }
    attr {
      key: "seed2"
      value {
        i: 0
      }
    }
  }
  node {
    name: "inference/fc1/weights/Initializer/random_uniform/sub"
    op: "Sub"
    input: "inference/fc1/weights/Initializer/random_uniform/max"
    input: "inference/fc1/weights/Initializer/random_uniform/min"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "inference/fc1/weights/Initializer/random_uniform/mul"
    op: "Mul"
    input: "inference/fc1/weights/Initializer/random_uniform/RandomUniform"
    input: "inference/fc1/weights/Initializer/random_uniform/sub"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc1/weights/Initializer/random_uniform"
    op: "Add"
    input: "inference/fc1/weights/Initializer/random_uniform/mul"
    input: "inference/fc1/weights/Initializer/random_uniform/min"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc1/weights/Assign"
    op: "Assign"
    input: "inference/fc1/weights"
    input: "inference/fc1/weights/Initializer/random_uniform"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/fc1/weights/read"
    op: "Identity"
    input: "inference/fc1/weights"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc1/MatMul"
    op: "MatMul"
    input: "inference/flatten/Reshape"
    input: "inference/fc1/weights/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: false
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: false
      }
    }
  }
  node {
    name: "inference/fc1/biases"
    op: "Variable"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 500
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "inference/fc1/biases/Initializer/zeros"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 500
            }
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "inference/fc1/biases/Assign"
    op: "Assign"
    input: "inference/fc1/biases"
    input: "inference/fc1/biases/Initializer/zeros"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/fc1/biases/read"
    op: "Identity"
    input: "inference/fc1/biases"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc1/BiasAdd"
    op: "BiasAdd"
    input: "inference/fc1/MatMul"
    input: "inference/fc1/biases/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "inference/fc1/Tanh"
    op: "Tanh"
    input: "inference/fc1/BiasAdd"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc2/Shape"
    op: "Shape"
    input: "inference/fc1/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "inference/fc2/unpack"
    op: "Unpack"
    input: "inference/fc2/Shape"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
          shape {
          }
        }
      }
    }
    attr {
      key: "axis"
      value {
        i: 0
      }
    }
    attr {
      key: "num"
      value {
        i: 2
      }
    }
  }
  node {
    name: "inference/fc2/weights"
    op: "Variable"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 500
          }
          dim {
            size: 1000
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "inference/fc2/weights/Initializer/random_uniform/shape"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 2
            }
          }
          tensor_content: "\364\001\000\000\350\003\000\000"
        }
      }
    }
  }
  node {
    name: "inference/fc2/weights/Initializer/random_uniform/min"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: -0.063245549798
        }
      }
    }
  }
  node {
    name: "inference/fc2/weights/Initializer/random_uniform/max"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.063245549798
        }
      }
    }
  }
  node {
    name: "inference/fc2/weights/Initializer/random_uniform/RandomUniform"
    op: "RandomUniform"
    input: "inference/fc2/weights/Initializer/random_uniform/shape"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "seed"
      value {
        i: 0
      }
    }
    attr {
      key: "seed2"
      value {
        i: 0
      }
    }
  }
  node {
    name: "inference/fc2/weights/Initializer/random_uniform/sub"
    op: "Sub"
    input: "inference/fc2/weights/Initializer/random_uniform/max"
    input: "inference/fc2/weights/Initializer/random_uniform/min"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "inference/fc2/weights/Initializer/random_uniform/mul"
    op: "Mul"
    input: "inference/fc2/weights/Initializer/random_uniform/RandomUniform"
    input: "inference/fc2/weights/Initializer/random_uniform/sub"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc2/weights/Initializer/random_uniform"
    op: "Add"
    input: "inference/fc2/weights/Initializer/random_uniform/mul"
    input: "inference/fc2/weights/Initializer/random_uniform/min"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc2/weights/Assign"
    op: "Assign"
    input: "inference/fc2/weights"
    input: "inference/fc2/weights/Initializer/random_uniform"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/fc2/weights/read"
    op: "Identity"
    input: "inference/fc2/weights"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc2/MatMul"
    op: "MatMul"
    input: "inference/fc1/Tanh"
    input: "inference/fc2/weights/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: false
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: false
      }
    }
  }
  node {
    name: "inference/fc2/biases"
    op: "Variable"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 1000
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "inference/fc2/biases/Initializer/zeros"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 1000
            }
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "inference/fc2/biases/Assign"
    op: "Assign"
    input: "inference/fc2/biases"
    input: "inference/fc2/biases/Initializer/zeros"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "inference/fc2/biases/read"
    op: "Identity"
    input: "inference/fc2/biases"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/fc2/BiasAdd"
    op: "BiasAdd"
    input: "inference/fc2/MatMul"
    input: "inference/fc2/biases/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "inference/fc2/Tanh"
    op: "Tanh"
    input: "inference/fc2/BiasAdd"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/predictions/Reshape/shape"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 2
            }
          }
          tensor_content: "\377\377\377\377\350\003\000\000"
        }
      }
    }
  }
  node {
    name: "inference/predictions/Reshape"
    op: "Reshape"
    input: "inference/fc2/Tanh"
    input: "inference/predictions/Reshape/shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/predictions/Softmax"
    op: "Softmax"
    input: "inference/predictions/Reshape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "inference/predictions/Shape"
    op: "Shape"
    input: "inference/fc2/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "inference/predictions/Reshape_1"
    op: "Reshape"
    input: "inference/predictions/Softmax"
    input: "inference/predictions/Shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "train/Placeholder"
    op: "Placeholder"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
        }
      }
    }
  }
  node {
    name: "train/Rank"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 2
        }
      }
    }
  }
  node {
    name: "train/Shape"
    op: "Shape"
    input: "inference/fc2/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "train/Rank_1"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 2
        }
      }
    }
  }
  node {
    name: "train/Shape_1"
    op: "Shape"
    input: "inference/fc2/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "train/Sub/y"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "train/Sub"
    op: "Sub"
    input: "train/Rank_1"
    input: "train/Sub/y"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "train/Slice/begin"
    op: "Pack"
    input: "train/Sub"
    attr {
      key: "N"
      value {
        i: 1
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "axis"
      value {
        i: 0
      }
    }
  }
  node {
    name: "train/Slice/size"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "train/Slice"
    op: "Slice"
    input: "train/Shape_1"
    input: "train/Slice/begin"
    input: "train/Slice/size"
    attr {
      key: "Index"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/concat/concat_dim"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 0
        }
      }
    }
  }
  node {
    name: "train/concat/values_0"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: -1
        }
      }
    }
  }
  node {
    name: "train/concat"
    op: "Concat"
    input: "train/concat/concat_dim"
    input: "train/concat/values_0"
    input: "train/Slice"
    attr {
      key: "N"
      value {
        i: 2
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
  }
  node {
    name: "train/Reshape"
    op: "Reshape"
    input: "inference/fc2/Tanh"
    input: "train/concat"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/Rank_2"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 2
        }
      }
    }
  }
  node {
    name: "train/Shape_2"
    op: "Shape"
    input: "train/Placeholder"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "train/Sub_1/y"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "train/Sub_1"
    op: "Sub"
    input: "train/Rank_2"
    input: "train/Sub_1/y"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "train/Slice_1/begin"
    op: "Pack"
    input: "train/Sub_1"
    attr {
      key: "N"
      value {
        i: 1
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "axis"
      value {
        i: 0
      }
    }
  }
  node {
    name: "train/Slice_1/size"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "train/Slice_1"
    op: "Slice"
    input: "train/Shape_2"
    input: "train/Slice_1/begin"
    input: "train/Slice_1/size"
    attr {
      key: "Index"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/concat_1/concat_dim"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 0
        }
      }
    }
  }
  node {
    name: "train/concat_1/values_0"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: -1
        }
      }
    }
  }
  node {
    name: "train/concat_1"
    op: "Concat"
    input: "train/concat_1/concat_dim"
    input: "train/concat_1/values_0"
    input: "train/Slice_1"
    attr {
      key: "N"
      value {
        i: 2
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
  }
  node {
    name: "train/Reshape_1"
    op: "Reshape"
    input: "train/Placeholder"
    input: "train/concat_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/loss"
    op: "SoftmaxCrossEntropyWithLogits"
    input: "train/Reshape"
    input: "train/Reshape_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/Sub_2/y"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "train/Sub_2"
    op: "Sub"
    input: "train/Rank"
    input: "train/Sub_2/y"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "train/Slice_2/begin"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: 0
        }
      }
    }
  }
  node {
    name: "train/Slice_2/size"
    op: "Pack"
    input: "train/Sub_2"
    attr {
      key: "N"
      value {
        i: 1
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "axis"
      value {
        i: 0
      }
    }
  }
  node {
    name: "train/Slice_2"
    op: "Slice"
    input: "train/Shape"
    input: "train/Slice_2/begin"
    input: "train/Slice_2/size"
    attr {
      key: "Index"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/Reshape_2"
    op: "Reshape"
    input: "train/loss"
    input: "train/Slice_2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/learning_rate"
    op: "Placeholder"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/Shape"
    op: "Shape"
    input: "train/Reshape_2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/Const"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 1.0
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/Fill"
    op: "Fill"
    input: "train/OptimizeLoss/gradients/Shape"
    input: "train/OptimizeLoss/gradients/Const"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/train/Reshape_2_grad/Shape"
    op: "Shape"
    input: "train/loss"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/train/Reshape_2_grad/Reshape"
    op: "Reshape"
    input: "train/OptimizeLoss/gradients/Fill"
    input: "train/OptimizeLoss/gradients/train/Reshape_2_grad/Shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/zeros_like"
    op: "ZerosLike"
    input: "train/loss:1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/train/loss_grad/ExpandDims/dim"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: -1
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/train/loss_grad/ExpandDims"
    op: "ExpandDims"
    input: "train/OptimizeLoss/gradients/train/Reshape_2_grad/Reshape"
    input: "train/OptimizeLoss/gradients/train/loss_grad/ExpandDims/dim"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tdim"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/train/loss_grad/mul"
    op: "Mul"
    input: "train/OptimizeLoss/gradients/train/loss_grad/ExpandDims"
    input: "train/loss:1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/train/Reshape_grad/Shape"
    op: "Shape"
    input: "inference/fc2/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/train/Reshape_grad/Reshape"
    op: "Reshape"
    input: "train/OptimizeLoss/gradients/train/loss_grad/mul"
    input: "train/OptimizeLoss/gradients/train/Reshape_grad/Shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
    op: "TanhGrad"
    input: "inference/fc2/Tanh"
    input: "train/OptimizeLoss/gradients/train/Reshape_grad/Reshape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/BiasAddGrad"
    op: "BiasAddGrad"
    input: "train/OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/group_deps"
    op: "NoOp"
    input: "^train/OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
    input: "^train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/BiasAddGrad"
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
    input: "^train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/BiasAddGrad"
    input: "^train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/BiasAddGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul"
    op: "MatMul"
    input: "train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency"
    input: "inference/fc2/weights/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: false
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul_1"
    op: "MatMul"
    input: "inference/fc1/Tanh"
    input: "train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: true
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/group_deps"
    op: "NoOp"
    input: "^train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul"
    input: "^train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul_1"
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/control_dependency"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul"
    input: "^train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul_1"
    input: "^train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul_1"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
    op: "TanhGrad"
    input: "inference/fc1/Tanh"
    input: "train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/BiasAddGrad"
    op: "BiasAddGrad"
    input: "train/OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/group_deps"
    op: "NoOp"
    input: "^train/OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
    input: "^train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/BiasAddGrad"
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
    input: "^train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/BiasAddGrad"
    input: "^train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/BiasAddGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul"
    op: "MatMul"
    input: "train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency"
    input: "inference/fc1/weights/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 156800
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: false
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul_1"
    op: "MatMul"
    input: "inference/flatten/Reshape"
    input: "train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: true
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/group_deps"
    op: "NoOp"
    input: "^train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul"
    input: "^train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul_1"
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/control_dependency"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul"
    input: "^train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 156800
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul_1"
    input: "^train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul_1"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/flatten/Reshape_grad/Shape"
    op: "Shape"
    input: "inference/conv2/MaxPool2D/MaxPool"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/flatten/Reshape_grad/Reshape"
    op: "Reshape"
    input: "train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/control_dependency"
    input: "train/OptimizeLoss/gradients/inference/flatten/Reshape_grad/Shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 56
            }
            dim {
              size: 56
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/MaxPool2D/MaxPool_grad/MaxPoolGrad"
    op: "MaxPoolGrad"
    input: "inference/conv2/Conv/Tanh"
    input: "inference/conv2/MaxPool2D/MaxPool"
    input: "train/OptimizeLoss/gradients/inference/flatten/Reshape_grad/Reshape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "ksize"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
    attr {
      key: "padding"
      value {
        s: "VALID"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
    op: "TanhGrad"
    input: "inference/conv2/Conv/Tanh"
    input: "train/OptimizeLoss/gradients/inference/conv2/MaxPool2D/MaxPool_grad/MaxPoolGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/BiasAddGrad"
    op: "BiasAddGrad"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/group_deps"
    op: "NoOp"
    input: "^train/OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
    input: "^train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/BiasAddGrad"
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
    input: "^train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/BiasAddGrad"
    input: "^train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/BiasAddGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Shape"
    op: "Shape"
    input: "inference/conv1/MaxPool2D/MaxPool"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropInput"
    op: "Conv2DBackpropInput"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Shape"
    input: "inference/conv2/Conv/weights/read"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Shape_1"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 4
            }
          }
          tensor_content: "\005\000\000\000\005\000\000\000\024\000\000\0002\000\000\000"
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropFilter"
    op: "Conv2DBackpropFilter"
    input: "inference/conv1/MaxPool2D/MaxPool"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Shape_1"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/group_deps"
    op: "NoOp"
    input: "^train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropInput"
    input: "^train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropFilter"
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/control_dependency"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropInput"
    input: "^train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropInput"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropFilter"
    input: "^train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropFilter"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/MaxPool2D/MaxPool_grad/MaxPoolGrad"
    op: "MaxPoolGrad"
    input: "inference/conv1/Conv/Tanh"
    input: "inference/conv1/MaxPool2D/MaxPool"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "ksize"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
    attr {
      key: "padding"
      value {
        s: "VALID"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
    op: "TanhGrad"
    input: "inference/conv1/Conv/Tanh"
    input: "train/OptimizeLoss/gradients/inference/conv1/MaxPool2D/MaxPool_grad/MaxPoolGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/BiasAddGrad"
    op: "BiasAddGrad"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/group_deps"
    op: "NoOp"
    input: "^train/OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
    input: "^train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/BiasAddGrad"
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
    input: "^train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/BiasAddGrad"
    input: "^train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/BiasAddGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Shape"
    op: "Shape"
    input: "input/Reshape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropInput"
    op: "Conv2DBackpropInput"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Shape"
    input: "inference/conv1/Conv/weights/read"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Shape_1"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 4
            }
          }
          tensor_content: "\005\000\000\000\005\000\000\000\003\000\000\000\024\000\000\000"
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropFilter"
    op: "Conv2DBackpropFilter"
    input: "input/Reshape"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Shape_1"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/group_deps"
    op: "NoOp"
    input: "^train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropInput"
    input: "^train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropFilter"
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/control_dependency"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropInput"
    input: "^train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropInput"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 3
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropFilter"
    input: "^train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropFilter"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/ScalarSummary/tags"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
          }
          string_val: "loss"
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/ScalarSummary"
    op: "ScalarSummary"
    input: "train/OptimizeLoss/ScalarSummary/tags"
    input: "train/Reshape_2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/Const"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 3
          }
          dim {
            size: 20
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad/Assign"
    op: "Assign"
    input: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    input: "train/OptimizeLoss/Const"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad/read"
    op: "Identity"
    input: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/Const_1"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 20
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 20
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad/Assign"
    op: "Assign"
    input: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    input: "train/OptimizeLoss/Const_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad/read"
    op: "Identity"
    input: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/Const_2"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 20
          }
          dim {
            size: 50
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad/Assign"
    op: "Assign"
    input: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    input: "train/OptimizeLoss/Const_2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad/read"
    op: "Identity"
    input: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/Const_3"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 50
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 50
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad/Assign"
    op: "Assign"
    input: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    input: "train/OptimizeLoss/Const_3"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad/read"
    op: "Identity"
    input: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/Const_4"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc1/weights/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 156800
          }
          dim {
            size: 500
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc1/weights/Adagrad/Assign"
    op: "Assign"
    input: "train/OptimizeLoss/inference/fc1/weights/Adagrad"
    input: "train/OptimizeLoss/Const_4"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc1/weights/Adagrad/read"
    op: "Identity"
    input: "train/OptimizeLoss/inference/fc1/weights/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/Const_5"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 500
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc1/biases/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 500
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc1/biases/Adagrad/Assign"
    op: "Assign"
    input: "train/OptimizeLoss/inference/fc1/biases/Adagrad"
    input: "train/OptimizeLoss/Const_5"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc1/biases/Adagrad/read"
    op: "Identity"
    input: "train/OptimizeLoss/inference/fc1/biases/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/Const_6"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc2/weights/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 500
          }
          dim {
            size: 1000
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc2/weights/Adagrad/Assign"
    op: "Assign"
    input: "train/OptimizeLoss/inference/fc2/weights/Adagrad"
    input: "train/OptimizeLoss/Const_6"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc2/weights/Adagrad/read"
    op: "Identity"
    input: "train/OptimizeLoss/inference/fc2/weights/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/Const_7"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 1000
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc2/biases/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 1000
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc2/biases/Adagrad/Assign"
    op: "Assign"
    input: "train/OptimizeLoss/inference/fc2/biases/Adagrad"
    input: "train/OptimizeLoss/Const_7"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "train/OptimizeLoss/inference/fc2/biases/Adagrad/read"
    op: "Identity"
    input: "train/OptimizeLoss/inference/fc2/biases/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "train/OptimizeLoss/train/update_inference/conv1/Conv/weights/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/conv1/Conv/weights"
    input: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    input: "train/learning_rate"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/train/update_inference/conv1/Conv/biases/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/conv1/Conv/biases"
    input: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    input: "train/learning_rate"
    input: "train/OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/train/update_inference/conv2/Conv/weights/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/conv2/Conv/weights"
    input: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    input: "train/learning_rate"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/train/update_inference/conv2/Conv/biases/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/conv2/Conv/biases"
    input: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    input: "train/learning_rate"
    input: "train/OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/train/update_inference/fc1/weights/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/fc1/weights"
    input: "train/OptimizeLoss/inference/fc1/weights/Adagrad"
    input: "train/learning_rate"
    input: "train/OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/train/update_inference/fc1/biases/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/fc1/biases"
    input: "train/OptimizeLoss/inference/fc1/biases/Adagrad"
    input: "train/learning_rate"
    input: "train/OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/train/update_inference/fc2/weights/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/fc2/weights"
    input: "train/OptimizeLoss/inference/fc2/weights/Adagrad"
    input: "train/learning_rate"
    input: "train/OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/train/update_inference/fc2/biases/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/fc2/biases"
    input: "train/OptimizeLoss/inference/fc2/biases/Adagrad"
    input: "train/learning_rate"
    input: "train/OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "train/OptimizeLoss/train"
    op: "NoOp"
    input: "^train/OptimizeLoss/train/update_inference/conv1/Conv/weights/ApplyAdagrad"
    input: "^train/OptimizeLoss/train/update_inference/conv1/Conv/biases/ApplyAdagrad"
    input: "^train/OptimizeLoss/train/update_inference/conv2/Conv/weights/ApplyAdagrad"
    input: "^train/OptimizeLoss/train/update_inference/conv2/Conv/biases/ApplyAdagrad"
    input: "^train/OptimizeLoss/train/update_inference/fc1/weights/ApplyAdagrad"
    input: "^train/OptimizeLoss/train/update_inference/fc1/biases/ApplyAdagrad"
    input: "^train/OptimizeLoss/train/update_inference/fc2/weights/ApplyAdagrad"
    input: "^train/OptimizeLoss/train/update_inference/fc2/biases/ApplyAdagrad"
  }
  node {
    name: "train/OptimizeLoss/control_dependency"
    op: "Identity"
    input: "train/Reshape_2"
    input: "^train/OptimizeLoss/train"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@train/Reshape_2"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "Rank"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 2
        }
      }
    }
  }
  node {
    name: "Shape"
    op: "Shape"
    input: "inference/fc2/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "Rank_1"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 2
        }
      }
    }
  }
  node {
    name: "Shape_1"
    op: "Shape"
    input: "inference/fc2/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "Sub/y"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "Sub"
    op: "Sub"
    input: "Rank_1"
    input: "Sub/y"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "Slice/begin"
    op: "Pack"
    input: "Sub"
    attr {
      key: "N"
      value {
        i: 1
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "axis"
      value {
        i: 0
      }
    }
  }
  node {
    name: "Slice/size"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "Slice"
    op: "Slice"
    input: "Shape_1"
    input: "Slice/begin"
    input: "Slice/size"
    attr {
      key: "Index"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
  }
  node {
    name: "concat/concat_dim"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 0
        }
      }
    }
  }
  node {
    name: "concat/values_0"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: -1
        }
      }
    }
  }
  node {
    name: "concat"
    op: "Concat"
    input: "concat/concat_dim"
    input: "concat/values_0"
    input: "Slice"
    attr {
      key: "N"
      value {
        i: 2
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
  }
  node {
    name: "Reshape"
    op: "Reshape"
    input: "inference/fc2/Tanh"
    input: "concat"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "Rank_2"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 2
        }
      }
    }
  }
  node {
    name: "Shape_2"
    op: "Shape"
    input: "train/Placeholder"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "Sub_1/y"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "Sub_1"
    op: "Sub"
    input: "Rank_2"
    input: "Sub_1/y"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "Slice_1/begin"
    op: "Pack"
    input: "Sub_1"
    attr {
      key: "N"
      value {
        i: 1
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "axis"
      value {
        i: 0
      }
    }
  }
  node {
    name: "Slice_1/size"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "Slice_1"
    op: "Slice"
    input: "Shape_2"
    input: "Slice_1/begin"
    input: "Slice_1/size"
    attr {
      key: "Index"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
  }
  node {
    name: "concat_1/concat_dim"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 0
        }
      }
    }
  }
  node {
    name: "concat_1/values_0"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: -1
        }
      }
    }
  }
  node {
    name: "concat_1"
    op: "Concat"
    input: "concat_1/concat_dim"
    input: "concat_1/values_0"
    input: "Slice_1"
    attr {
      key: "N"
      value {
        i: 2
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
  }
  node {
    name: "Reshape_1"
    op: "Reshape"
    input: "train/Placeholder"
    input: "concat_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "loss"
    op: "SoftmaxCrossEntropyWithLogits"
    input: "Reshape"
    input: "Reshape_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "Sub_2/y"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 1
        }
      }
    }
  }
  node {
    name: "Sub_2"
    op: "Sub"
    input: "Rank"
    input: "Sub_2/y"
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "Slice_2/begin"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 1
            }
          }
          int_val: 0
        }
      }
    }
  }
  node {
    name: "Slice_2/size"
    op: "Pack"
    input: "Sub_2"
    attr {
      key: "N"
      value {
        i: 1
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "axis"
      value {
        i: 0
      }
    }
  }
  node {
    name: "Slice_2"
    op: "Slice"
    input: "Shape"
    input: "Slice_2/begin"
    input: "Slice_2/size"
    attr {
      key: "Index"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "T"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "Reshape_2"
    op: "Reshape"
    input: "loss"
    input: "Slice_2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "learning_rate"
    op: "Placeholder"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/Shape"
    op: "Shape"
    input: "Reshape_2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/Const"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 1.0
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/Fill"
    op: "Fill"
    input: "OptimizeLoss/gradients/Shape"
    input: "OptimizeLoss/gradients/Const"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/Reshape_2_grad/Shape"
    op: "Shape"
    input: "loss"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/Reshape_2_grad/Reshape"
    op: "Reshape"
    input: "OptimizeLoss/gradients/Fill"
    input: "OptimizeLoss/gradients/Reshape_2_grad/Shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/zeros_like"
    op: "ZerosLike"
    input: "loss:1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/loss_grad/ExpandDims/dim"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: -1
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/loss_grad/ExpandDims"
    op: "ExpandDims"
    input: "OptimizeLoss/gradients/Reshape_2_grad/Reshape"
    input: "OptimizeLoss/gradients/loss_grad/ExpandDims/dim"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tdim"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/loss_grad/mul"
    op: "Mul"
    input: "OptimizeLoss/gradients/loss_grad/ExpandDims"
    input: "loss:1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/Reshape_grad/Shape"
    op: "Shape"
    input: "inference/fc2/Tanh"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/Reshape_grad/Reshape"
    op: "Reshape"
    input: "OptimizeLoss/gradients/loss_grad/mul"
    input: "OptimizeLoss/gradients/Reshape_grad/Shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
    op: "TanhGrad"
    input: "inference/fc2/Tanh"
    input: "OptimizeLoss/gradients/Reshape_grad/Reshape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/BiasAddGrad"
    op: "BiasAddGrad"
    input: "OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/group_deps"
    op: "NoOp"
    input: "^OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
    input: "^OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/BiasAddGrad"
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
    input: "^OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/fc2/Tanh_grad/TanhGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/BiasAddGrad"
    input: "^OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/BiasAddGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul"
    op: "MatMul"
    input: "OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency"
    input: "inference/fc2/weights/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: false
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul_1"
    op: "MatMul"
    input: "inference/fc1/Tanh"
    input: "OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: true
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/group_deps"
    op: "NoOp"
    input: "^OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul"
    input: "^OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul_1"
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/control_dependency"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul"
    input: "^OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul_1"
    input: "^OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/fc2/MatMul_grad/MatMul_1"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
    op: "TanhGrad"
    input: "inference/fc1/Tanh"
    input: "OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/BiasAddGrad"
    op: "BiasAddGrad"
    input: "OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/group_deps"
    op: "NoOp"
    input: "^OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
    input: "^OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/BiasAddGrad"
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
    input: "^OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/fc1/Tanh_grad/TanhGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/BiasAddGrad"
    input: "^OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/BiasAddGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul"
    op: "MatMul"
    input: "OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency"
    input: "inference/fc1/weights/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 156800
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: false
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul_1"
    op: "MatMul"
    input: "inference/flatten/Reshape"
    input: "OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: true
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/group_deps"
    op: "NoOp"
    input: "^OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul"
    input: "^OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul_1"
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/control_dependency"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul"
    input: "^OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 156800
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul_1"
    input: "^OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/fc1/MatMul_grad/MatMul_1"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/flatten/Reshape_grad/Shape"
    op: "Shape"
    input: "inference/conv2/MaxPool2D/MaxPool"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/flatten/Reshape_grad/Reshape"
    op: "Reshape"
    input: "OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/control_dependency"
    input: "OptimizeLoss/gradients/inference/flatten/Reshape_grad/Shape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tshape"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 56
            }
            dim {
              size: 56
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/MaxPool2D/MaxPool_grad/MaxPoolGrad"
    op: "MaxPoolGrad"
    input: "inference/conv2/Conv/Tanh"
    input: "inference/conv2/MaxPool2D/MaxPool"
    input: "OptimizeLoss/gradients/inference/flatten/Reshape_grad/Reshape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "ksize"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
    attr {
      key: "padding"
      value {
        s: "VALID"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
    op: "TanhGrad"
    input: "inference/conv2/Conv/Tanh"
    input: "OptimizeLoss/gradients/inference/conv2/MaxPool2D/MaxPool_grad/MaxPoolGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/BiasAddGrad"
    op: "BiasAddGrad"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/group_deps"
    op: "NoOp"
    input: "^OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
    input: "^OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/BiasAddGrad"
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
    input: "^OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/conv2/Conv/Tanh_grad/TanhGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/BiasAddGrad"
    input: "^OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/BiasAddGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Shape"
    op: "Shape"
    input: "inference/conv1/MaxPool2D/MaxPool"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropInput"
    op: "Conv2DBackpropInput"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Shape"
    input: "inference/conv2/Conv/weights/read"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Shape_1"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 4
            }
          }
          tensor_content: "\005\000\000\000\005\000\000\000\024\000\000\0002\000\000\000"
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropFilter"
    op: "Conv2DBackpropFilter"
    input: "inference/conv1/MaxPool2D/MaxPool"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Shape_1"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/group_deps"
    op: "NoOp"
    input: "^OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropInput"
    input: "^OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropFilter"
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/control_dependency"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropInput"
    input: "^OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropInput"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 112
            }
            dim {
              size: 112
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropFilter"
    input: "^OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/Conv2DBackpropFilter"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/MaxPool2D/MaxPool_grad/MaxPoolGrad"
    op: "MaxPoolGrad"
    input: "inference/conv1/Conv/Tanh"
    input: "inference/conv1/MaxPool2D/MaxPool"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "ksize"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
    attr {
      key: "padding"
      value {
        s: "VALID"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 2
          i: 2
          i: 1
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
    op: "TanhGrad"
    input: "inference/conv1/Conv/Tanh"
    input: "OptimizeLoss/gradients/inference/conv1/MaxPool2D/MaxPool_grad/MaxPoolGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/BiasAddGrad"
    op: "BiasAddGrad"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/group_deps"
    op: "NoOp"
    input: "^OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
    input: "^OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/BiasAddGrad"
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
    input: "^OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/conv1/Conv/Tanh_grad/TanhGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/BiasAddGrad"
    input: "^OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/BiasAddGrad"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Shape"
    op: "Shape"
    input: "input/Reshape"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "out_type"
      value {
        type: DT_INT32
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropInput"
    op: "Conv2DBackpropInput"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Shape"
    input: "inference/conv1/Conv/weights/read"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: -1
            }
            dim {
              size: -1
            }
            dim {
              size: -1
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Shape_1"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 4
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 4
            }
          }
          tensor_content: "\005\000\000\000\005\000\000\000\003\000\000\000\024\000\000\000"
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropFilter"
    op: "Conv2DBackpropFilter"
    input: "input/Reshape"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Shape_1"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "data_format"
      value {
        s: "NHWC"
      }
    }
    attr {
      key: "padding"
      value {
        s: "SAME"
      }
    }
    attr {
      key: "strides"
      value {
        list {
          i: 1
          i: 1
          i: 1
          i: 1
        }
      }
    }
    attr {
      key: "use_cudnn_on_gpu"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/group_deps"
    op: "NoOp"
    input: "^OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropInput"
    input: "^OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropFilter"
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/control_dependency"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropInput"
    input: "^OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropInput"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 224
            }
            dim {
              size: 224
            }
            dim {
              size: 3
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/control_dependency_1"
    op: "Identity"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropFilter"
    input: "^OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/group_deps"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/Conv2DBackpropFilter"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/ScalarSummary/tags"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
          }
          string_val: "loss"
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/ScalarSummary"
    op: "ScalarSummary"
    input: "OptimizeLoss/ScalarSummary/tags"
    input: "Reshape_2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/Const"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 3
          }
          dim {
            size: 20
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad/Assign"
    op: "Assign"
    input: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    input: "OptimizeLoss/Const"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad/read"
    op: "Identity"
    input: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/Const_1"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 20
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 20
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad/Assign"
    op: "Assign"
    input: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    input: "OptimizeLoss/Const_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad/read"
    op: "Identity"
    input: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/Const_2"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 20
          }
          dim {
            size: 50
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad/Assign"
    op: "Assign"
    input: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    input: "OptimizeLoss/Const_2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad/read"
    op: "Identity"
    input: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/Const_3"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 50
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 50
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad/Assign"
    op: "Assign"
    input: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    input: "OptimizeLoss/Const_3"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad/read"
    op: "Identity"
    input: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/Const_4"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc1/weights/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 156800
          }
          dim {
            size: 500
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc1/weights/Adagrad/Assign"
    op: "Assign"
    input: "OptimizeLoss/inference/fc1/weights/Adagrad"
    input: "OptimizeLoss/Const_4"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc1/weights/Adagrad/read"
    op: "Identity"
    input: "OptimizeLoss/inference/fc1/weights/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/Const_5"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 500
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc1/biases/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 500
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc1/biases/Adagrad/Assign"
    op: "Assign"
    input: "OptimizeLoss/inference/fc1/biases/Adagrad"
    input: "OptimizeLoss/Const_5"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc1/biases/Adagrad/read"
    op: "Identity"
    input: "OptimizeLoss/inference/fc1/biases/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/Const_6"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc2/weights/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 500
          }
          dim {
            size: 1000
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc2/weights/Adagrad/Assign"
    op: "Assign"
    input: "OptimizeLoss/inference/fc2/weights/Adagrad"
    input: "OptimizeLoss/Const_6"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc2/weights/Adagrad/read"
    op: "Identity"
    input: "OptimizeLoss/inference/fc2/weights/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/Const_7"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 1000
            }
          }
          float_val: 0.10000000149
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc2/biases/Adagrad"
    op: "Variable"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 1000
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc2/biases/Adagrad/Assign"
    op: "Assign"
    input: "OptimizeLoss/inference/fc2/biases/Adagrad"
    input: "OptimizeLoss/Const_7"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "OptimizeLoss/inference/fc2/biases/Adagrad/read"
    op: "Identity"
    input: "OptimizeLoss/inference/fc2/biases/Adagrad"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
  }
  node {
    name: "OptimizeLoss/train/update_inference/conv1/Conv/weights/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/conv1/Conv/weights"
    input: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    input: "learning_rate"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/Conv2D_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/train/update_inference/conv1/Conv/biases/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/conv1/Conv/biases"
    input: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    input: "learning_rate"
    input: "OptimizeLoss/gradients/inference/conv1/Conv/BiasAdd_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/train/update_inference/conv2/Conv/weights/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/conv2/Conv/weights"
    input: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    input: "learning_rate"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/Conv2D_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/train/update_inference/conv2/Conv/biases/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/conv2/Conv/biases"
    input: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    input: "learning_rate"
    input: "OptimizeLoss/gradients/inference/conv2/Conv/BiasAdd_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/train/update_inference/fc1/weights/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/fc1/weights"
    input: "OptimizeLoss/inference/fc1/weights/Adagrad"
    input: "learning_rate"
    input: "OptimizeLoss/gradients/inference/fc1/MatMul_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/train/update_inference/fc1/biases/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/fc1/biases"
    input: "OptimizeLoss/inference/fc1/biases/Adagrad"
    input: "learning_rate"
    input: "OptimizeLoss/gradients/inference/fc1/BiasAdd_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/train/update_inference/fc2/weights/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/fc2/weights"
    input: "OptimizeLoss/inference/fc2/weights/Adagrad"
    input: "learning_rate"
    input: "OptimizeLoss/gradients/inference/fc2/MatMul_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/train/update_inference/fc2/biases/ApplyAdagrad"
    op: "ApplyAdagrad"
    input: "inference/fc2/biases"
    input: "OptimizeLoss/inference/fc2/biases/Adagrad"
    input: "learning_rate"
    input: "OptimizeLoss/gradients/inference/fc2/BiasAdd_grad/tuple/control_dependency_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "OptimizeLoss/train"
    op: "NoOp"
    input: "^OptimizeLoss/train/update_inference/conv1/Conv/weights/ApplyAdagrad"
    input: "^OptimizeLoss/train/update_inference/conv1/Conv/biases/ApplyAdagrad"
    input: "^OptimizeLoss/train/update_inference/conv2/Conv/weights/ApplyAdagrad"
    input: "^OptimizeLoss/train/update_inference/conv2/Conv/biases/ApplyAdagrad"
    input: "^OptimizeLoss/train/update_inference/fc1/weights/ApplyAdagrad"
    input: "^OptimizeLoss/train/update_inference/fc1/biases/ApplyAdagrad"
    input: "^OptimizeLoss/train/update_inference/fc2/weights/ApplyAdagrad"
    input: "^OptimizeLoss/train/update_inference/fc2/biases/ApplyAdagrad"
  }
  node {
    name: "OptimizeLoss/control_dependency"
    op: "Identity"
    input: "Reshape_2"
    input: "^OptimizeLoss/train"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@Reshape_2"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
          }
        }
      }
    }
  }
  node {
    name: "MergeSummary/MergeSummary"
    op: "MergeSummary"
    input: "train/OptimizeLoss/ScalarSummary"
    input: "OptimizeLoss/ScalarSummary"
    attr {
      key: "N"
      value {
        i: 2
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "save/Const"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
          }
          string_val: "model"
        }
      }
    }
  }
  node {
    name: "save/save/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 24
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 24
            }
          }
          string_val: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
          string_val: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
          string_val: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
          string_val: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
          string_val: "OptimizeLoss/inference/fc1/biases/Adagrad"
          string_val: "OptimizeLoss/inference/fc1/weights/Adagrad"
          string_val: "OptimizeLoss/inference/fc2/biases/Adagrad"
          string_val: "OptimizeLoss/inference/fc2/weights/Adagrad"
          string_val: "inference/conv1/Conv/biases"
          string_val: "inference/conv1/Conv/weights"
          string_val: "inference/conv2/Conv/biases"
          string_val: "inference/conv2/Conv/weights"
          string_val: "inference/fc1/biases"
          string_val: "inference/fc1/weights"
          string_val: "inference/fc2/biases"
          string_val: "inference/fc2/weights"
          string_val: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
          string_val: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
          string_val: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
          string_val: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
          string_val: "train/OptimizeLoss/inference/fc1/biases/Adagrad"
          string_val: "train/OptimizeLoss/inference/fc1/weights/Adagrad"
          string_val: "train/OptimizeLoss/inference/fc2/biases/Adagrad"
          string_val: "train/OptimizeLoss/inference/fc2/weights/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/save/shapes_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 24
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 24
            }
          }
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/save"
    op: "SaveSlices"
    input: "save/Const"
    input: "save/save/tensor_names"
    input: "save/save/shapes_and_slices"
    input: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    input: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    input: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    input: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    input: "OptimizeLoss/inference/fc1/biases/Adagrad"
    input: "OptimizeLoss/inference/fc1/weights/Adagrad"
    input: "OptimizeLoss/inference/fc2/biases/Adagrad"
    input: "OptimizeLoss/inference/fc2/weights/Adagrad"
    input: "inference/conv1/Conv/biases"
    input: "inference/conv1/Conv/weights"
    input: "inference/conv2/Conv/biases"
    input: "inference/conv2/Conv/weights"
    input: "inference/fc1/biases"
    input: "inference/fc1/weights"
    input: "inference/fc2/biases"
    input: "inference/fc2/weights"
    input: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    input: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    input: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    input: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    input: "train/OptimizeLoss/inference/fc1/biases/Adagrad"
    input: "train/OptimizeLoss/inference/fc1/weights/Adagrad"
    input: "train/OptimizeLoss/inference/fc2/biases/Adagrad"
    input: "train/OptimizeLoss/inference/fc2/weights/Adagrad"
    attr {
      key: "T"
      value {
        list {
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/control_dependency"
    op: "Identity"
    input: "save/Const"
    input: "^save/save"
    attr {
      key: "T"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@save/Const"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "save/RestoreV2/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2/tensor_names"
    input: "save/RestoreV2/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign"
    op: "Assign"
    input: "OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    input: "save/RestoreV2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_1/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_1/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_1"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_1/tensor_names"
    input: "save/RestoreV2_1/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_1"
    op: "Assign"
    input: "OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    input: "save/RestoreV2_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_2/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_2/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_2"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_2/tensor_names"
    input: "save/RestoreV2_2/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_2"
    op: "Assign"
    input: "OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    input: "save/RestoreV2_2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_3/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_3/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_3"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_3/tensor_names"
    input: "save/RestoreV2_3/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_3"
    op: "Assign"
    input: "OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    input: "save/RestoreV2_3"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_4/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "OptimizeLoss/inference/fc1/biases/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_4/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_4"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_4/tensor_names"
    input: "save/RestoreV2_4/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_4"
    op: "Assign"
    input: "OptimizeLoss/inference/fc1/biases/Adagrad"
    input: "save/RestoreV2_4"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_5/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "OptimizeLoss/inference/fc1/weights/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_5/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_5"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_5/tensor_names"
    input: "save/RestoreV2_5/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_5"
    op: "Assign"
    input: "OptimizeLoss/inference/fc1/weights/Adagrad"
    input: "save/RestoreV2_5"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_6/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "OptimizeLoss/inference/fc2/biases/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_6/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_6"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_6/tensor_names"
    input: "save/RestoreV2_6/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_6"
    op: "Assign"
    input: "OptimizeLoss/inference/fc2/biases/Adagrad"
    input: "save/RestoreV2_6"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_7/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "OptimizeLoss/inference/fc2/weights/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_7/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_7"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_7/tensor_names"
    input: "save/RestoreV2_7/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_7"
    op: "Assign"
    input: "OptimizeLoss/inference/fc2/weights/Adagrad"
    input: "save/RestoreV2_7"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_8/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "inference/conv1/Conv/biases"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_8/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_8"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_8/tensor_names"
    input: "save/RestoreV2_8/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_8"
    op: "Assign"
    input: "inference/conv1/Conv/biases"
    input: "save/RestoreV2_8"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_9/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "inference/conv1/Conv/weights"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_9/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_9"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_9/tensor_names"
    input: "save/RestoreV2_9/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_9"
    op: "Assign"
    input: "inference/conv1/Conv/weights"
    input: "save/RestoreV2_9"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_10/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "inference/conv2/Conv/biases"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_10/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_10"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_10/tensor_names"
    input: "save/RestoreV2_10/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_10"
    op: "Assign"
    input: "inference/conv2/Conv/biases"
    input: "save/RestoreV2_10"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_11/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "inference/conv2/Conv/weights"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_11/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_11"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_11/tensor_names"
    input: "save/RestoreV2_11/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_11"
    op: "Assign"
    input: "inference/conv2/Conv/weights"
    input: "save/RestoreV2_11"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_12/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "inference/fc1/biases"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_12/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_12"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_12/tensor_names"
    input: "save/RestoreV2_12/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_12"
    op: "Assign"
    input: "inference/fc1/biases"
    input: "save/RestoreV2_12"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_13/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "inference/fc1/weights"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_13/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_13"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_13/tensor_names"
    input: "save/RestoreV2_13/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_13"
    op: "Assign"
    input: "inference/fc1/weights"
    input: "save/RestoreV2_13"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_14/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "inference/fc2/biases"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_14/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_14"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_14/tensor_names"
    input: "save/RestoreV2_14/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_14"
    op: "Assign"
    input: "inference/fc2/biases"
    input: "save/RestoreV2_14"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_15/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "inference/fc2/weights"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_15/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_15"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_15/tensor_names"
    input: "save/RestoreV2_15/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_15"
    op: "Assign"
    input: "inference/fc2/weights"
    input: "save/RestoreV2_15"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_16/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_16/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_16"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_16/tensor_names"
    input: "save/RestoreV2_16/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_16"
    op: "Assign"
    input: "train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad"
    input: "save/RestoreV2_16"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_17/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_17/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_17"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_17/tensor_names"
    input: "save/RestoreV2_17/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_17"
    op: "Assign"
    input: "train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad"
    input: "save/RestoreV2_17"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv1/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 3
            }
            dim {
              size: 20
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_18/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_18/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_18"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_18/tensor_names"
    input: "save/RestoreV2_18/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_18"
    op: "Assign"
    input: "train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad"
    input: "save/RestoreV2_18"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_19/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_19/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_19"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_19/tensor_names"
    input: "save/RestoreV2_19/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_19"
    op: "Assign"
    input: "train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad"
    input: "save/RestoreV2_19"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/conv2/Conv/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 5
            }
            dim {
              size: 5
            }
            dim {
              size: 20
            }
            dim {
              size: 50
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_20/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "train/OptimizeLoss/inference/fc1/biases/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_20/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_20"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_20/tensor_names"
    input: "save/RestoreV2_20/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_20"
    op: "Assign"
    input: "train/OptimizeLoss/inference/fc1/biases/Adagrad"
    input: "save/RestoreV2_20"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_21/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "train/OptimizeLoss/inference/fc1/weights/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_21/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_21"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_21/tensor_names"
    input: "save/RestoreV2_21/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_21"
    op: "Assign"
    input: "train/OptimizeLoss/inference/fc1/weights/Adagrad"
    input: "save/RestoreV2_21"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc1/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 156800
            }
            dim {
              size: 500
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_22/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "train/OptimizeLoss/inference/fc2/biases/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_22/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_22"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_22/tensor_names"
    input: "save/RestoreV2_22/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_22"
    op: "Assign"
    input: "train/OptimizeLoss/inference/fc2/biases/Adagrad"
    input: "save/RestoreV2_22"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/biases"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_23/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "train/OptimizeLoss/inference/fc2/weights/Adagrad"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_23/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_23"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_23/tensor_names"
    input: "save/RestoreV2_23/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_23"
    op: "Assign"
    input: "train/OptimizeLoss/inference/fc2/weights/Adagrad"
    input: "save/RestoreV2_23"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@inference/fc2/weights"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 500
            }
            dim {
              size: 1000
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/restore_all"
    op: "NoOp"
    input: "^save/Assign"
    input: "^save/Assign_1"
    input: "^save/Assign_2"
    input: "^save/Assign_3"
    input: "^save/Assign_4"
    input: "^save/Assign_5"
    input: "^save/Assign_6"
    input: "^save/Assign_7"
    input: "^save/Assign_8"
    input: "^save/Assign_9"
    input: "^save/Assign_10"
    input: "^save/Assign_11"
    input: "^save/Assign_12"
    input: "^save/Assign_13"
    input: "^save/Assign_14"
    input: "^save/Assign_15"
    input: "^save/Assign_16"
    input: "^save/Assign_17"
    input: "^save/Assign_18"
    input: "^save/Assign_19"
    input: "^save/Assign_20"
    input: "^save/Assign_21"
    input: "^save/Assign_22"
    input: "^save/Assign_23"
  }
  versions {
    producer: 15
  }
}
saver_def {
  filename_tensor_name: "save/Const:0"
  save_tensor_name: "save/control_dependency:0"
  restore_op_name: "save/restore_all"
  max_to_keep: 5
  keep_checkpoint_every_n_hours: 10000.0
  version: V1
}
collection_def {
  key: "model_variables"
  value {
    node_list {
      value: "inference/conv1/Conv/weights:0"
      value: "inference/conv1/Conv/biases:0"
      value: "inference/conv2/Conv/weights:0"
      value: "inference/conv2/Conv/biases:0"
      value: "inference/fc1/weights:0"
      value: "inference/fc1/biases:0"
      value: "inference/fc2/weights:0"
      value: "inference/fc2/biases:0"
    }
  }
}
collection_def {
  key: "summaries"
  value {
    node_list {
      value: "train/OptimizeLoss/ScalarSummary:0"
      value: "OptimizeLoss/ScalarSummary:0"
    }
  }
}
collection_def {
  key: "trainable_variables"
  value {
    bytes_list {
      value: "\n\036inference/conv1/Conv/weights:0\022#inference/conv1/Conv/weights/Assign\032#inference/conv1/Conv/weights/read:0"
      value: "\n\035inference/conv1/Conv/biases:0\022\"inference/conv1/Conv/biases/Assign\032\"inference/conv1/Conv/biases/read:0"
      value: "\n\036inference/conv2/Conv/weights:0\022#inference/conv2/Conv/weights/Assign\032#inference/conv2/Conv/weights/read:0"
      value: "\n\035inference/conv2/Conv/biases:0\022\"inference/conv2/Conv/biases/Assign\032\"inference/conv2/Conv/biases/read:0"
      value: "\n\027inference/fc1/weights:0\022\034inference/fc1/weights/Assign\032\034inference/fc1/weights/read:0"
      value: "\n\026inference/fc1/biases:0\022\033inference/fc1/biases/Assign\032\033inference/fc1/biases/read:0"
      value: "\n\027inference/fc2/weights:0\022\034inference/fc2/weights/Assign\032\034inference/fc2/weights/read:0"
      value: "\n\026inference/fc2/biases:0\022\033inference/fc2/biases/Assign\032\033inference/fc2/biases/read:0"
    }
  }
}
collection_def {
  key: "variables"
  value {
    bytes_list {
      value: "\n\036inference/conv1/Conv/weights:0\022#inference/conv1/Conv/weights/Assign\032#inference/conv1/Conv/weights/read:0"
      value: "\n\035inference/conv1/Conv/biases:0\022\"inference/conv1/Conv/biases/Assign\032\"inference/conv1/Conv/biases/read:0"
      value: "\n\036inference/conv2/Conv/weights:0\022#inference/conv2/Conv/weights/Assign\032#inference/conv2/Conv/weights/read:0"
      value: "\n\035inference/conv2/Conv/biases:0\022\"inference/conv2/Conv/biases/Assign\032\"inference/conv2/Conv/biases/read:0"
      value: "\n\027inference/fc1/weights:0\022\034inference/fc1/weights/Assign\032\034inference/fc1/weights/read:0"
      value: "\n\026inference/fc1/biases:0\022\033inference/fc1/biases/Assign\032\033inference/fc1/biases/read:0"
      value: "\n\027inference/fc2/weights:0\022\034inference/fc2/weights/Assign\032\034inference/fc2/weights/read:0"
      value: "\n\026inference/fc2/biases:0\022\033inference/fc2/biases/Assign\032\033inference/fc2/biases/read:0"
      value: "\n9train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad:0\022>train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad/Assign\032>train/OptimizeLoss/inference/conv1/Conv/weights/Adagrad/read:0"
      value: "\n8train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad:0\022=train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad/Assign\032=train/OptimizeLoss/inference/conv1/Conv/biases/Adagrad/read:0"
      value: "\n9train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad:0\022>train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad/Assign\032>train/OptimizeLoss/inference/conv2/Conv/weights/Adagrad/read:0"
      value: "\n8train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad:0\022=train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad/Assign\032=train/OptimizeLoss/inference/conv2/Conv/biases/Adagrad/read:0"
      value: "\n2train/OptimizeLoss/inference/fc1/weights/Adagrad:0\0227train/OptimizeLoss/inference/fc1/weights/Adagrad/Assign\0327train/OptimizeLoss/inference/fc1/weights/Adagrad/read:0"
      value: "\n1train/OptimizeLoss/inference/fc1/biases/Adagrad:0\0226train/OptimizeLoss/inference/fc1/biases/Adagrad/Assign\0326train/OptimizeLoss/inference/fc1/biases/Adagrad/read:0"
      value: "\n2train/OptimizeLoss/inference/fc2/weights/Adagrad:0\0227train/OptimizeLoss/inference/fc2/weights/Adagrad/Assign\0327train/OptimizeLoss/inference/fc2/weights/Adagrad/read:0"
      value: "\n1train/OptimizeLoss/inference/fc2/biases/Adagrad:0\0226train/OptimizeLoss/inference/fc2/biases/Adagrad/Assign\0326train/OptimizeLoss/inference/fc2/biases/Adagrad/read:0"
      value: "\n3OptimizeLoss/inference/conv1/Conv/weights/Adagrad:0\0228OptimizeLoss/inference/conv1/Conv/weights/Adagrad/Assign\0328OptimizeLoss/inference/conv1/Conv/weights/Adagrad/read:0"
      value: "\n2OptimizeLoss/inference/conv1/Conv/biases/Adagrad:0\0227OptimizeLoss/inference/conv1/Conv/biases/Adagrad/Assign\0327OptimizeLoss/inference/conv1/Conv/biases/Adagrad/read:0"
      value: "\n3OptimizeLoss/inference/conv2/Conv/weights/Adagrad:0\0228OptimizeLoss/inference/conv2/Conv/weights/Adagrad/Assign\0328OptimizeLoss/inference/conv2/Conv/weights/Adagrad/read:0"
      value: "\n2OptimizeLoss/inference/conv2/Conv/biases/Adagrad:0\0227OptimizeLoss/inference/conv2/Conv/biases/Adagrad/Assign\0327OptimizeLoss/inference/conv2/Conv/biases/Adagrad/read:0"
      value: "\n,OptimizeLoss/inference/fc1/weights/Adagrad:0\0221OptimizeLoss/inference/fc1/weights/Adagrad/Assign\0321OptimizeLoss/inference/fc1/weights/Adagrad/read:0"
      value: "\n+OptimizeLoss/inference/fc1/biases/Adagrad:0\0220OptimizeLoss/inference/fc1/biases/Adagrad/Assign\0320OptimizeLoss/inference/fc1/biases/Adagrad/read:0"
      value: "\n,OptimizeLoss/inference/fc2/weights/Adagrad:0\0221OptimizeLoss/inference/fc2/weights/Adagrad/Assign\0321OptimizeLoss/inference/fc2/weights/Adagrad/read:0"
      value: "\n+OptimizeLoss/inference/fc2/biases/Adagrad:0\0220OptimizeLoss/inference/fc2/biases/Adagrad/Assign\0320OptimizeLoss/inference/fc2/biases/Adagrad/read:0"
    }
  }
}
