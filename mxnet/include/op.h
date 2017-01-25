/*!
*  Copyright (c) 2016 by Contributors
* \file op.h
* \brief definition of all the operators
* \author Chuntao Hong
*/

#ifndef _MXNETOP_H
#define _MXNETOP_H

#include <string>
#include <vector>
#include "base.h"
#include "shape.h"
#include "MxNetCpp.h"

namespace mxnet {
namespace cpp {

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_add(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_sub(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_mul(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_div(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reshape input according to a target shape spec.
 *        The target shape is a tuple and can be a simple list of dimensions
 *        such as (12,3) or it can incorporate special codes that correspond
 *        The special codes are all expressed as integers less than 1. These
 *        codes effectively refer to a machine that pops input dims off the
 *        beginning of the input dims list and pushes resulting output dims
 *        onto the end of the output dims list, which starts empty. The codes
 *        0  Copy     Pop one input dim and push it onto the output dims
 *        -1  Infer    Push a dim that is inferred later from all other output
 *        -2  CopyAll  Pop all remaining input dims and push them onto output
 *        -3  Merge2   Pop two input dims, multiply them, and push result
 *        -4  Split2   Pop one input dim, and read two next target shape specs,
 *        push them both onto output dims (either can be -1 and will
 *        be inferred from the other
 *        The exact mathematical behavior of these codes is given in the
 *        description of the 'shape' parameter. All non-codes (positive
 *        integers) just pop a dim off the input dims (if any), throw it away,
 *        Examples:
 *        Type     Input      Target            Output
 *        Copy     (2,3,4)    (4,0,2)           (4,3,2)
 *        Copy     (2,3,4)    (2,0,0)           (2,3,4)
 *        Infer    (2,3,4)    (6,1,-1)          (6,1,4)
 *        Infer    (2,3,4)    (3,-1,8)          (3,1,8)
 *        CopyAll  (9,8,7)    (-2)              (9,8,7)
 *        CopyAll  (9,8,7)    (9,-2)            (9,8,7)
 *        CopyAll  (9,8,7)    (-2,1,1)          (9,8,7,1,1)
 *        Merge2   (3,4)      (-3)              (12)
 *        Merge2   (3,4,5)    (-3,0)            (12,5)
 *        Merge2   (3,4,5)    (0,-3)            (3,20)
 *        Merge2   (3,4,5,6)  (-3,0,0)          (12,5,6)
 *        Merge2   (3,4,5,6)  (-3,-2)           (12,5,6)
 *        Split2   (12)       (-4,6,2)          (6,2)
 *        Split2   (12)       (-4,2,6)          (2,6)
 *        Split2   (12)       (-4,-1,6)         (2,6)
 *        Split2   (12,9)     (-4,2,6,0)        (2,6,9)
 *        Split2   (12,9,9,9) (-4,2,6,-2)       (2,6,9,9,9)
 *        Split2   (12,12)    (-4,2,-1,-4,-1,2) (2,6,6,2)
 *
 *
 *        From:src/operator/tensor/matrix_op.cc:61
 * \param symbol_name name of the resulting symbol
 * \param data Input data to reshape.
 * \param target_shape (Deprecated! Use shape instead.) Target new shape. One
 *        and only one dim can be 0, in which case it will be inferred from
 * \param keep_highest (Deprecated! Use shape instead.) Whether keep the
 *        highest dim unchanged.If set to true, then the first dim in
 * \param shape Target shape, a tuple, t=(t_1,t_2,..,t_m).
 *        Let the input dims be s=(s_1,s_2,..,s_n).
 *        The output dims u=(u_1,u_2,..,u_p) are computed from s and t.
 *        The target shape tuple elements t_i are read in order, and used to
 *        t_i:       meaning:      behavior:
 *        +ve        explicit      u_p = t_i
 *        0          copy          u_p = s_i
 *        -1         infer         u_p = (Prod s_i) / (Prod u_j | j != p)
 *        -2         copy all      u_p = s_i, u_p+1 = s_i+1, ...
 *        -3         merge two     u_p = s_i * s_i+1
 *        -4,a,b     split two     u_p = a, u_p+1 = b | a * b = s_i
 *        The split directive (-4) in the target shape tuple is followed by two
 *        dimensions, one of which can be -1, which means it will be inferred
 *        The can only be one globally inferred dimension (-1), aside from any
 * \param reverse Whether to match the shapes from the backward. If reverse is
 *        true, 0 values in the `shape` argument will be searched from the
 *        backward. E.g the original shape is (10, 5, 4) and the shape
 *        argument is (-1, 0). If reverse is true, the new shape should be
 * \return new symbol
 */
inline Symbol Reshape(const std::string& symbol_name,
                      Symbol data,
                      Shape target_shape = Shape(0,0),
                      bool keep_highest = false,
                      Shape shape = Shape(),
                      bool reverse = false) {
  return Operator("Reshape")
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Flatten input into 2D by collapsing all the higher dimensions.
 *        A (d1, d2, ..., dK) tensor is flatten to (d1, d2* ... *dK) matrix.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to reshape.
 * \return new symbol
 */
inline Symbol Flatten(const std::string& symbol_name,
                      Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Transpose the input tensor and return a new one
 *
 *        From:src/operator/tensor/matrix_op.cc:93
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(const std::string& symbol_name,
                        Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Expand the shape of array by inserting a new axis.
 *
 *        From:src/operator/tensor/matrix_op.cc:121
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Position (amongst axes) where new axis is to be inserted.
 * \return new symbol
 */
inline Symbol expand_dims(const std::string& symbol_name,
                          Symbol data,
                          int axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif (Crop the input tensor and return a new one.
 *
 *        Requirements
 *        ------------
 *        - the input and output (if explicitly given) are of the same data
 *        and on the same device.
 *        )
 *
 *        From:src/operator/tensor/matrix_op.cc:142
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param begin starting coordinates
 * \param end ending coordinates
 * \return new symbol
 */
inline Symbol crop(const std::string& symbol_name,
                   Symbol data,
                   Shape begin,
                   Shape end) {
  return Operator("crop")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Slice the input along certain axis and return a sliced array.
 *
 *        From:src/operator/tensor/matrix_op.cc:197
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis The axis to be sliced
 * \param begin The beginning index to be sliced
 * \param end The end index to be sliced
 * \return new symbol
 */
inline Symbol slice_axis(const std::string& symbol_name,
                         Symbol data,
                         int axis,
                         int begin,
                         int end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Flip the input tensor along axis and return a new one.
 *
 *        From:src/operator/tensor/matrix_op.cc:216
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis The dimension to flip
 * \return new symbol
 */
inline Symbol flip(const std::string& symbol_name,
                   Symbol data,
                   int axis) {
  return Operator("flip")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate dot product of two matrices or two vectors.
 *
 *        From:src/operator/tensor/matrix_op.cc:228
 * \param symbol_name name of the resulting symbol
 * \param lhs Left input
 * \param rhs Right input
 * \param transpose_a True if the first matrix is transposed.
 * \param transpose_b True if the second matrix is tranposed.
 * \return new symbol
 */
inline Symbol dot(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false) {
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate batched dot product of two matrices. (batch, M, K)
 *
 *        From:src/operator/tensor/matrix_op.cc:254
 * \param symbol_name name of the resulting symbol
 * \param lhs Left input
 * \param rhs Right input
 * \param axis The dimension to flip
 * \return new symbol
 */
inline Symbol batch_dot(const std::string& symbol_name,
                        Symbol lhs,
                        Symbol rhs,
                        int axis) {
  return Operator("batch_dot")
           .SetParam("axis", axis)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(const std::string& symbol_name,
                           Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_power(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_maximum(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_minimum(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_hypot(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_hypot")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute argmax
 *
 *        From:src/operator/tensor/broadcast_reduce_op_index.cc:11
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Empty or unsigned. The axis to perform the reduction.If left
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol argmax(const std::string& symbol_name,
                     Symbol data,
                     int axis = -1,
                     bool keepdims = false) {
  return Operator("argmax")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute argmin
 *
 *        From:src/operator/tensor/broadcast_reduce_op_index.cc:15
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Empty or unsigned. The axis to perform the reduction.If left
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol argmin(const std::string& symbol_name,
                     Symbol data,
                     int axis = -1,
                     bool keepdims = false) {
  return Operator("argmin")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param src Source input
 * \return new symbol
 */
inline Symbol argmax_channel(const std::string& symbol_name,
                             Symbol src) {
  return Operator("argmax_channel")
           .SetInput("src", src)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Sum src along axis. If axis is empty, global reduction is performed
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:17
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol sum(const std::string& symbol_name,
                  Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("sum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute product of src along axis. If axis is empty, global reduction
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:27
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol prod(const std::string& symbol_name,
                   Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false) {
  return Operator("prod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Sum src along axis, ignoring NaN values. If axis is empty, global
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:37
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol nansum(const std::string& symbol_name,
                     Symbol data,
                     Shape axis = Shape(),
                     bool keepdims = false) {
  return Operator("nansum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute product of src along axis, ignoring NaN values. If axis is
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:47
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol nanprod(const std::string& symbol_name,
                      Symbol data,
                      Shape axis = Shape(),
                      bool keepdims = false) {
  return Operator("nanprod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute max along axis. If axis is empty, global reduction is
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:57
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol max(const std::string& symbol_name,
                  Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("max")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute min along axis. If axis is empty, global reduction is
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:67
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol min(const std::string& symbol_name,
                  Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("min")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Broadcast src along axis
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:76
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis The axes to perform the broadcasting.
 * \param size Target sizes of the broadcasting axes.
 * \return new symbol
 */
inline Symbol broadcast_axis(const std::string& symbol_name,
                             Symbol data,
                             Shape axis = Shape(),
                             Shape size = Shape()) {
  return Operator("broadcast_axis")
           .SetParam("axis", axis)
           .SetParam("size", size)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Broadcast src to shape
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:83
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param shape The shape of the desired array. We can set the dim to zero if
 *        it's same as the original. E.g `A = broadcast_to(B, shape=(10, 0,
 *        0))` has the same meaning as `A = broadcast_axis(B, axis=0,
 * \return new symbol
 */
inline Symbol broadcast_to(const std::string& symbol_name,
                           Symbol data,
                           Shape shape = Shape()) {
  return Operator("broadcast_to")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param src Source input
 * \return new symbol
 */
inline Symbol norm(const std::string& symbol_name,
                   Symbol src) {
  return Operator("norm")
           .SetInput("src", src)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Negate src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:52
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol negative(const std::string& symbol_name,
                       Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take absolute value of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:58
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol abs(const std::string& symbol_name,
                  Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take sign of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:67
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol sign(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take round of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:76
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol round(const std::string& symbol_name,
                    Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take ceil of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:81
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol ceil(const std::string& symbol_name,
                   Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take floor of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:86
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol floor(const std::string& symbol_name,
                    Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take round of the src to nearest integer
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:91
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol rint(const std::string& symbol_name,
                   Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take round of the src to integer nearest 0
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:96
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol fix(const std::string& symbol_name,
                  Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take square of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:101
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol square(const std::string& symbol_name,
                     Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take square root of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:110
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol sqrt(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take reciprocal square root of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:119
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol rsqrt(const std::string& symbol_name,
                    Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take exp of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:129
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol exp(const std::string& symbol_name,
                  Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take log of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:135
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol log(const std::string& symbol_name,
                  Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take base-10 log of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:141
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol log10(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take base-2 log of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:147
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol log2(const std::string& symbol_name,
                   Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take sin of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:156
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol sin(const std::string& symbol_name,
                  Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take `log(1 + x)` in a numerically stable way
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:165
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol log1p(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take `exp(x) - 1` in a numerically stable way
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:174
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol expm1(const std::string& symbol_name,
                    Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take cos of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:183
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol cos(const std::string& symbol_name,
                  Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take tan of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:192
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol tan(const std::string& symbol_name,
                  Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take arcsin of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:201
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol arcsin(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take arccos of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:210
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol arccos(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take arctan of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:219
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol arctan(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take degrees of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:228
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol degrees(const std::string& symbol_name,
                      Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take radians of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:237
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol radians(const std::string& symbol_name,
                      Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take sinh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:246
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol sinh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take cosh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:255
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol cosh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take tanh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:264
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol tanh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take arcsinh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:273
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol arcsinh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take arccosh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:282
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol arccosh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take arctanh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:291
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol arctanh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take the gamma function (extension of the factorial function) of the
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:300
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol gamma(const std::string& symbol_name,
                    Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take gammaln (log of the absolute value of gamma(x)) of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:309
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol gammaln(const std::string& symbol_name,
                      Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate Smooth L1 Loss(lhs, scalar)
 *
 *        From:src/operator/tensor/elemwise_binary_scalar_op_extended.cc:63
 * \param symbol_name name of the resulting symbol
 * \param data source input
 * \param scalar scalar input
 * \return new symbol
 */
inline Symbol smooth_l1(const std::string& symbol_name,
                        Symbol data,
                        mx_float scalar) {
  return Operator("smooth_l1")
           .SetParam("scalar", scalar)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Map integer index to vector representations (embeddings). Those
 *        embeddings are learnable parameters. For a input of shape (d1, ...,
 *        dK), the output shape is (d1, ..., dK, output_dim). All the input
 *
 *        From:src/operator/tensor/indexing_op.cc:17
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the EmbeddingOp.
 * \param weight Embedding weight matrix.
 * \param input_dim vocabulary size of the input indices.
 * \param output_dim dimension of the embedding vectors.
 * \return new symbol
 */
inline Symbol Embedding(const std::string& symbol_name,
                        Symbol data,
                        Symbol weight,
                        int input_dim,
                        int output_dim) {
  return Operator("Embedding")
           .SetParam("input_dim", input_dim)
           .SetParam("output_dim", output_dim)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_equal(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_not_equal(const std::string& symbol_name,
                                  Symbol lhs,
                                  Symbol rhs) {
  return Operator("broadcast_not_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_greater(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_greater")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_greater_equal(const std::string& symbol_name,
                                      Symbol lhs,
                                      Symbol rhs) {
  return Operator("broadcast_greater_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_lesser(const std::string& symbol_name,
                               Symbol lhs,
                               Symbol rhs) {
  return Operator("broadcast_lesser")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_lesser_equal(const std::string& symbol_name,
                                     Symbol lhs,
                                     Symbol rhs) {
  return Operator("broadcast_lesser_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Sample a uniform distribution
 * \param symbol_name name of the resulting symbol
 * \param low The lower bound of distribution
 * \param high The upper bound of distribution
 * \param shape The shape of the output
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n).Only used
 * \return new symbol
 */
inline Symbol uniform(const std::string& symbol_name,
                      mx_float low = 0,
                      mx_float high = 1,
                      Shape shape = Shape(),
                      const std::string& ctx = "") {
  return Operator("uniform")
           .SetParam("low", low)
           .SetParam("high", high)
           .SetParam("shape", shape)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Sample a normal distribution
 * \param symbol_name name of the resulting symbol
 * \param loc Mean of the distribution.
 * \param scale Standard deviation of the distribution.
 * \param shape The shape of the output
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n).Only used
 * \return new symbol
 */
inline Symbol normal(const std::string& symbol_name,
                     mx_float loc = 0,
                     mx_float scale = 1,
                     Shape shape = Shape(),
                     const std::string& ctx = "") {
  return Operator("normal")
           .SetParam("loc", loc)
           .SetParam("scale", scale)
           .SetParam("shape", shape)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Perform element sum of inputs
 *
 *        From:src/operator/tensor/elemwise_sum.cc:56
 * \param symbol_name name of the resulting symbol
 * \param args List of input tensors
 * \return new symbol
 */
inline Symbol ElementWiseSum(const std::string& symbol_name,
                             const std::vector<Symbol>& args) {
  return Operator("ElementWiseSum")
(args)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Updater function for sgd optimizer
 * \param symbol_name name of the resulting symbol
 * \return new symbol
 */
inline Symbol sgd_update(const std::string& symbol_name) {
  return Operator("sgd_update")
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Updater function for sgd optimizer
 * \param symbol_name name of the resulting symbol
 * \return new symbol
 */
inline Symbol sgd_mom_update(const std::string& symbol_name) {
  return Operator("sgd_mom_update")
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Updater function for adam optimizer
 * \param symbol_name name of the resulting symbol
 * \return new symbol
 */
inline Symbol adam_update(const std::string& symbol_name) {
  return Operator("adam_update")
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate cross_entropy(lhs, one_hot(rhs))
 *
 *        From:src/operator/loss_binary_op.cc:12
 * \param symbol_name name of the resulting symbol
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(const std::string& symbol_name,
                                    Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif An operator taking in a n-dimensional input tensor (n > 2), and
 *        normalizing the input by subtracting the mean and variance
 *        calculated over the spatial dimensions. This is an implemention of
 *        the operator described in "Instance Normalization: The Missing
 *        Ingredient for Fast Stylization", D. Ulyanov, A. Vedaldi, V.
 *        Lempitsky, 2016 (arXiv:1607.08022v2). This layer is similar to batch
 *        normalization, with two differences: first, the normalization is
 *        carried out per example ('instance'), not over a batch. Second, the
 *        same normalization is applied both at test and train time. This
 * \param symbol_name name of the resulting symbol
 * \param data A n-dimensional tensor (n > 2) of the form [batch, channel,
 * \param gamma A vector of length 'channel', which multiplies the normalized
 * \param beta A vector of length 'channel', which is added to the product of
 * \param eps Epsilon to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(const std::string& symbol_name,
                           Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Support Vector Machine based transformation on input, backprop L2-SVM
 * \param symbol_name name of the resulting symbol
 * \param data Input data to svm.
 * \param label Label data.
 * \param margin Scale the DType(param_.margin) for activation size
 * \param regularization_coefficient Scale the coefficient responsible for
 * \param use_linear If set true, uses L1-SVM objective function. Default uses
 * \return new symbol
 */
inline Symbol SVMOutput(const std::string& symbol_name,
                        Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif the type of RNN to compute
 */
enum class RNNMode {
  gru = 0,
  lstm = 1,
  rnn_relu = 2,
  rnn_tanh = 3
};

/*!
 * \breif Apply a recurrent layer to input.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p Dropout probability, fraction of the input that gets dropped out at
 * \param state_outputs Whether to have the states as symbol outputs.
 * \return new symbol
 */
inline Symbol RNN(const std::string& symbol_name,
                  Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  int state_size,
                  int num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .CreateSymbol(symbol_name);
}

/*! \breif Target data type.
 */
enum class CastDtype {
  float16 = 0,
  float32 = 1,
  float64 = 2,
  int32 = 3,
  uint8 = 4
};

/*!
 * \breif Cast array to a different data type.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to cast function.
 * \param dtype Target data type.
 * \return new symbol
 */
inline Symbol Cast(const std::string& symbol_name,
                   Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Crop the 2nd and 3rd dim of input data, with the corresponding size
 *        of h_w or with width and height of the second input symbol, i.e.,
 *        with one input, we need h_w to specify the crop height and width,
 * \param symbol_name name of the resulting symbol
 * \param data Tensor or List of Tensors, the second input will be used as
 * \param num_args Number of inputs for crop, if equals one, then we will use
 *        the h_wfor crop height and width, else if equals two, then we will
 *        use the heightand width of the second input symbol, we name
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and weight: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it
 * \return new symbol
 */
inline Symbol Crop(const std::string& symbol_name,
                   Symbol data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reverses the elements of each sequence. Takes an n-dimensional tensor
 *        of the form [max sequence length, batchsize, other dims] and returns
 *        a tensor of the same shape. This operator takes an optional input
 *        tensor sequence_length of positive ints of dimension [batchsize]
 *        when the sequence_length option is set to true. This allows the
 *        operator to handle variable-length sequences. If sequence_length is
 *        false, then each example in the batch is assumed to have the max
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input tensor of the form [max sequence length,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \return new symbol
 */
inline Symbol SequenceReverse(const std::string& symbol_name,
                              Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*! \breif transformation type
 */
enum class SpatialTransformerTransformType {
  affine = 0
};

/*! \breif sampling type
 */
enum class SpatialTransformerSamplerType {
  bilinear = 0
};

/*!
 * \breif Apply spatial transformer to input feature map.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type
 *        is affine, and the name of loc symbol should better starts with
 *        'stn_loc', so that initialization it with iddentify tranform, or you
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \return new symbol
 */
inline Symbol SpatialTransformer(const std::string& symbol_name,
                                 Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0)) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply swapaxis to input.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the SwapAxisOp.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(const std::string& symbol_name,
                       Symbol data,
                       int dim1 = 0,
                       int dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Padding type to use. "constant" pads all values with a constant
 *        value, the value of which can be specified with the constant_value
 */
enum class PadMode {
  constant = 0,
  edge = 1
};

/*!
 * \breif Pads an n-dimensional input tensor. Allows for precise control of the
 *        padding type and how much padding to apply on both sides of a given
 * \param symbol_name name of the resulting symbol
 * \param data An n-dimensional input tensor.
 * \param mode Padding type to use. "constant" pads all values with a constant
 *        value, the value of which can be specified with the constant_value
 * \param pad_width A tuple of padding widths of length 2*r, where r is the
 *        rank of the input tensor, specifying number of values padded to the
 *        edges of each axis. (before_1, after_1, ... , before_N, after_N)
 *        unique pad widths for each axis. Equivalent to pad_width in
 * \param constant_value This option is only used when mode is "constant". This
 *        value will be used as the padding value. Defaults to 0 if not
 * \return new symbol
 */
inline Symbol Pad(const std::string& symbol_name,
                  Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif If set to null, op will do nothing on output gradient.If set to
 *        batch, op will normalize gradient by divide batch sizeIf set to
 */
enum class SoftmaxOutputNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif Perform a softmax transformation on input, backprop with logloss.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to softmax.
 * \param label Label data, can also be probability value with same shape as
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the label value will be ignored during backward (only
 * \param multi_output If set to true, for a (n,k,x_1,..,x_n) dimensional input
 *        tensor, softmax will generate n*x_1*...*x_n output, each has k
 * \param use_ignore If set to true, the ignore_label value will not contribute
 * \param preserve_shape If true, for a (n_1, n_2, ..., n_d, k) dimensional
 *        input tensor, softmax will generate (n1, n2, ..., n_d, k) output,
 * \param normalization If set to null, op will do nothing on output
 *        gradient.If set to batch, op will normalize gradient by divide batch
 *        sizeIf set to valid, op will normalize gradient by divide sample not
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol SoftmaxOutput(const std::string& symbol_name,
                            Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::null,
                            bool out_grad = false) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif If set to null, op will do nothing on output gradient.If set to
 *        batch, op will normalize gradient by divide batch sizeIf set to
 */
enum class SoftmaxNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif DEPRECATED: Perform a softmax transformation on input. Please use
 * \param symbol_name name of the resulting symbol
 * \param data Input data to softmax.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the label value will be ignored during backward (only
 * \param multi_output If set to true, for a (n,k,x_1,..,x_n) dimensional input
 *        tensor, softmax will generate n*x_1*...*x_n output, each has k
 * \param use_ignore If set to true, the ignore_label value will not contribute
 * \param preserve_shape If true, for a (n_1, n_2, ..., n_d, k) dimensional
 *        input tensor, softmax will generate (n1, n2, ..., n_d, k) output,
 * \param normalization If set to null, op will do nothing on output
 *        gradient.If set to batch, op will normalize gradient by divide batch
 *        sizeIf set to valid, op will normalize gradient by divide sample not
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol Softmax(const std::string& symbol_name,
                      Symbol data,
                      mx_float grad_scale = 1,
                      mx_float ignore_label = -1,
                      bool multi_output = false,
                      bool use_ignore = false,
                      bool preserve_shape = false,
                      SoftmaxNormalization normalization = SoftmaxNormalization::null,
                      bool out_grad = false) {
  static const char *SoftmaxNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("Softmax")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Activation function to be applied.
 */
enum class LeakyReLUActType {
  elu = 0,
  leaky = 1,
  prelu = 2,
  rrelu = 3
};

/*!
 * \breif Apply activation function to input.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(const std::string& symbol_name,
                        Symbol data,
                        LeakyReLUActType act_type = LeakyReLUActType::leaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.334) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "leaky",
    "prelu",
    "rrelu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Takes the last element of a sequence. Takes an n-dimensional tensor
 *        of the form [max sequence length, batchsize, other dims] and returns
 *        a (n-1)-dimensional tensor of the form [batchsize, other dims]. This
 *        operator takes an optional input tensor sequence_length of positive
 *        ints of dimension [batchsize] when the sequence_length option is set
 *        to true. This allows the operator to handle variable-length
 *        sequences. If sequence_length is false, then each example in the
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input tensor of the form [max sequence length,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \return new symbol
 */
inline Symbol SequenceLast(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*! \breif Normalization Mode. If set to instance, this operator will compute a
 *        norm for each instance in the batch; this is the default mode. If
 *        set to channel, this operator will compute a cross channel norm at
 *        each position of each instance. If set to spatial, this operator
 */
enum class L2NormalizationMode {
  channel = 0,
  instance = 1,
  spatial = 2
};

/*!
 * \breif Set the l2 norm of each instance to a constant.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the L2NormalizationOp.
 * \param eps Epsilon to prevent div 0
 * \param mode Normalization Mode. If set to instance, this operator will
 *        compute a norm for each instance in the batch; this is the default
 *        mode. If set to channel, this operator will compute a cross channel
 *        norm at each position of each instance. If set to spatial, this
 * \return new symbol
 */
inline Symbol L2Normalization(const std::string& symbol_name,
                              Symbol data,
                              mx_float eps = 1e-10,
                              L2NormalizationMode mode = L2NormalizationMode::instance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply convolution to input then add a bias.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionOp.
 * \param nsize normalization window width in elements.
 * \param alpha value of the alpha variance scaling parameter in the
 * \param beta value of the beta power parameter in the normalization formula
 * \param knorm value of the k parameter in normalization formula
 * \return new symbol
 */
inline Symbol LRN(const std::string& symbol_name,
                  Symbol data,
                  int nsize,
                  mx_float alpha = 0.0001,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply correlation to inputs
 * \param symbol_name name of the resulting symbol
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(const std::string& symbol_name,
                          Symbol data1,
                          Symbol data2,
                          int kernel_size = 1,
                          int max_displacement = 1,
                          int stride1 = 1,
                          int stride2 = 1,
                          int pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Sets all elements outside the sequence to a constant value. Takes an
 *        n-dimensional tensor of the form [max sequence length, batchsize,
 *        other dims] and returns a tensor of the same shape. This operator
 *        takes an optional input tensor sequence_length of positive ints of
 *        dimension [batchsize] when the sequence_length option is set to
 *        true. This allows the operator to handle variable-length sequences.
 *        If sequence_length is false, then each example in the batch is
 *        assumed to have the max sequence length, and this operator becomes
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input tensor of the form [max sequence length,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \param value The value to be used as a mask.
 * \return new symbol
 */
inline Symbol SequenceMask(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Get output from a symbol and pass 0 gradient back
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \return new symbol
 */
inline Symbol BlockGrad(const std::string& symbol_name,
                        Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply a sparse regularization to the output a sigmoid activation
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(const std::string& symbol_name,
                                        Symbol data,
                                        mx_float sparseness_target = 0.1,
                                        mx_float penalty = 0.001,
                                        mx_float momentum = 0.9) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif upsampling method
 */
enum class UpSamplingSampleType {
  bilinear = 0,
  nearest = 1
};

/*! \breif How to handle multiple input. concat means concatenate upsampled
 *        images along the channel dimension. sum means add all images
 */
enum class UpSamplingMultiInputMode {
  concat = 0,
  sum = 1
};

/*!
 * \breif Perform nearest neighboor/bilinear up sampling to inputs
 * \param symbol_name name of the resulting symbol
 * \param data Array of tensors to upsample
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor
 *        upsampling, this can be 1-N; the size of output will
 *        be(scale*h_0,scale*w_0) and all other inputs will be upsampled to
 *        thesame size. For bilinear upsampling this must be 2; 1 input and 1
 * \param num_filter Input filter. Only used by bilinear sample_type.
 * \param multi_input_mode How to handle multiple input. concat means
 *        concatenate upsampled images along the channel dimension. sum means
 *        add all images together, only available for nearest neighbor
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::string& symbol_name,
                         const std::vector<Symbol>& data,
                         int scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         int num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::concat,
                         int64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply deconvolution to input then add a bias.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the DeconvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel deconvolution kernel size: (y, x)
 * \param num_filter deconvolution filter(channel) number
 * \param stride deconvolution stride: (y, x)
 * \param pad pad for deconvolution: (y, x), a good number is : (kernel-1)/2,
 *        if target_shape set, pad will be ignored and will be computed
 * \param adj adjustment for output shape: (y, x), if target_shape set, adj
 * \param target_shape output shape with targe shape : (y, x)
 * \param num_group number of groups partition
 * \param workspace Tmp workspace for deconvolution (MB)
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol Deconvolution(const std::string& symbol_name,
                            Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            int num_filter,
                            Shape stride = Shape(1,1),
                            Shape pad = Shape(0,0),
                            Shape adj = Shape(0,0),
                            Shape target_shape = Shape(0,0),
                            int num_group = 1,
                            int64_t workspace = 512,
                            bool no_bias = true) {
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*! \breif Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 */
enum class ConvolutionCudnnTune {
  None = 0,
  fastest = 1,
  limited_workspace = 2,
  off = 3
};

/*! \breif Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 */
enum class ConvolutionLayout {
  None = 0,
  NCDHW = 1,
  NCHW = 2,
  NDHWC = 3,
  NHWC = 4
};

/*!
 * \breif Apply convolution to input then add a bias.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions. Equivalent to slicing input
 *        partitions, apply convolution on each, then concatenate the results
 * \param workspace Maximum tmp workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution(const std::string& symbol_name,
                          Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          int num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          int num_group = 1,
                          int64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::None,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::None) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs region-of-interest pooling on inputs. Resize bounding box
 *        coordinates by spatial_scale and crop input feature maps
 *        accordingly. The cropped feature maps are pooled by max pooling to a
 *        fixed size output indicated by pooled_size. batch_size will change
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator, a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1,
 *        x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners
 *        of designated region of interest. batch_index indicates the index of
 * \param pooled_size fix pooled size: (h, w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image
 *        height (or w). Equals the reciprocal of total stride in
 * \return new symbol
 */
inline Symbol ROIPooling(const std::string& symbol_name,
                         Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply batch normalization to input.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to batch normalization
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of
 *        local batch-norm. This will force change batch-norm into a scale
 * \return new symbol
 */
inline Symbol CuDNNBatchNorm(const std::string& symbol_name,
                             Symbol data,
                             mx_float eps = 0.001,
                             mx_float momentum = 0.9,
                             bool fix_gamma = true,
                             bool use_global_stats = false) {
  return Operator("CuDNNBatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply matrix multiplication to input then add a bias.
 *        It maps the input of shape `(batch_size, input_dim)` to the shape of
 *        `(batch_size, num_hidden)`. Learnable parameters include the weights
 *        of the linear transform and an optional bias vector.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the FullyConnectedOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol FullyConnected(const std::string& symbol_name,
                             Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Use linear regression for final output, this is used on final output
 * \param symbol_name name of the resulting symbol
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(const std::string& symbol_name,
                                     Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Use mean absolute error regression for final output, this is used on
 * \param symbol_name name of the resulting symbol
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(const std::string& symbol_name,
                                  Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Use Logistic regression for final output, this is used on final
 *        Logistic regression is suitable for binary classification or
 * \param symbol_name name of the resulting symbol
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(const std::string& symbol_name,
                                       Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif If set to null, op will not normalize on output gradient.If set to
 *        batch, op will normalize gradient by divide batch size.If set to
 */
enum class MakeLossNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif Get output from a symbol and pass 1 gradient back. This is used as a
 *        terminal loss if unary and binary operator are used to composite a
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param grad_scale gradient scale as a supplement to unary and binary
 * \param valid_thresh regard element valid when x > valid_thresh, this is used
 * \param normalization If set to null, op will not normalize on output
 *        gradient.If set to batch, op will normalize gradient by divide batch
 *        size.If set to valid, op will normalize gradient by divide # sample
 * \return new symbol
 */
inline Symbol MakeLoss(const std::string& symbol_name,
                       Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::null) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Slice input equally along specified axis
 * \param symbol_name name of the resulting symbol
 * \param num_outputs Number of outputs to be sliced.
 * \param axis Dimension along which to slice.
 * \param squeeze_axis If true AND the sliced dimension becomes 1, squeeze that
 * \return new symbol
 */
inline Symbol SliceChannel(const std::string& symbol_name,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Perform a feature concat on channel dim (defaut is 1) over all
 * \param symbol_name name of the resulting symbol
 * \param data List of tensors to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::string& symbol_name,
                     const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol(symbol_name);
}

/*! \breif Activation function to be applied.
 */
enum class ActivationActType {
  relu = 0,
  sigmoid = 1,
  softrelu = 2,
  tanh = 3
};

/*!
 * \breif Elementwise activation function.
 *
 *        The following activation types are supported (operations are applied
 *        scalar of the input tensor):
 *
 *        - `relu`: Rectified Linear Unit, `y = max(x, 0)`
 *        - `sigmoid`: `y = 1 / (1 + exp(-x))`
 *        - `tanh`: Hyperbolic tangent, `y = (exp(x) - exp(-x)) / (exp(x) +
 *        - `softrelu`: Soft ReLU, or SoftPlus, `y = log(1 + exp(x))`
 *
 *        See `LeakyReLU` for other activations with parameters.
 *
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(const std::string& symbol_name,
                         Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Custom operator implemented in frontend.
 * \param symbol_name name of the resulting symbol
 * \param op_type Type of custom operator. Must be registered first.
 * \return new symbol
 */
inline Symbol Custom(const std::string& symbol_name,
                     const std::string& op_type) {
  return Operator("Custom")
           .CreateSymbol(symbol_name);
}

/*! \breif Pooling type to be applied.
 */
enum class PoolingPoolType {
  avg = 0,
  max = 1,
  sum = 2
};

/*! \breif Pooling convention to be applied.kValid is default setting of Mxnet
 *        and rounds down the output pooling size.kFull is compatible with
 */
enum class PoolingPoolingConvention {
  full = 0,
  valid = 1
};

/*!
 * \breif Perform spatial pooling on inputs.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current
 * \param pooling_convention Pooling convention to be applied.kValid is default
 *        setting of Mxnet and rounds down the output pooling size.kFull is
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling(const std::string& symbol_name,
                      Symbol data,
                      Shape kernel,
                      PoolingPoolType pool_type,
                      bool global_pool = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::valid,
                      Shape stride = Shape(1,1),
                      Shape pad = Shape(0,0)) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply batch normalization to input.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to batch normalization
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of
 *        local batch-norm. This will force change batch-norm into a scale
 * \return new symbol
 */
inline Symbol BatchNorm(const std::string& symbol_name,
                        Symbol data,
                        mx_float eps = 0.001,
                        mx_float momentum = 0.9,
                        bool fix_gamma = true,
                        bool use_global_stats = false) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply dropout to input.
 *        During training, each element of the input is randomly set to zero
 *        And then the whole tensor is rescaled by 1/(1-p) to keep the
 *        before applying dropout. During the test time, this behaves as an
 *
 * \param symbol_name name of the resulting symbol
 * \param data Input data to dropout.
 * \param p Fraction of the input that gets dropped out at training time
 * \return new symbol
 */
inline Symbol Dropout(const std::string& symbol_name,
                      Symbol data,
                      mx_float p = 0.5) {
  return Operator("Dropout")
           .SetParam("p", p)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Softmax Mode. If set to instance, this operator will compute a
 *        softmax for each instance in the batch; this is the default mode. If
 *        set to channel, this operator will compute a num_channel-class
 *        softmax at each position of each instance; this can be used for
 */
enum class SoftmaxActivationMode {
  channel = 0,
  instance = 1
};

/*!
 * \breif Apply softmax activation to input. This is intended for internal
 *        layers. For output (loss layer) please use SoftmaxOutput. If
 *        mode=instance, this operator will compute a softmax for each
 *        instance in the batch; this is the default mode. If mode=channel,
 *        this operator will compute a num_channel-class softmax at each
 *        position of each instance; this can be used for fully convolutional
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param mode Softmax Mode. If set to instance, this operator will compute a
 *        softmax for each instance in the batch; this is the default mode. If
 *        set to channel, this operator will compute a num_channel-class
 *        softmax at each position of each instance; this can be used for
 * \return new symbol
 */
inline Symbol SoftmaxActivation(const std::string& symbol_name,
                                Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::instance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Choose one element from each line(row for python, column for R/Julia)
 *        in lhs according to index indicated by rhs. This function assume rhs
 * \param symbol_name name of the resulting symbol
 * \param lhs Left operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol choose_element_0index(const std::string& symbol_name,
                                    Symbol lhs,
                                    Symbol rhs) {
  return Operator("choose_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Fill one element of each line(row for python, column for R/Julia) in
 *        lhs according to index indicated by rhs and values indicated by mhs.
 * \param symbol_name name of the resulting symbol
 * \param lhs Left operand to the function.
 * \param mhs Middle operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol fill_element_0index(const std::string& symbol_name,
                                  Symbol lhs,
                                  Symbol mhs,
                                  Symbol rhs) {
  return Operator("fill_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("mhs", mhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Clip ndarray elements to range (a_min, a_max)
 * \param symbol_name name of the resulting symbol
 * \param src Source input
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(const std::string& symbol_name,
                   Symbol src,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("src", src)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_add(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_sub(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_mul(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_div(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Reshape input according to a target shape spec.
 *        The target shape is a tuple and can be a simple list of dimensions
 *        such as (12,3) or it can incorporate special codes that correspond
 *        The special codes are all expressed as integers less than 1. These
 *        codes effectively refer to a machine that pops input dims off the
 *        beginning of the input dims list and pushes resulting output dims
 *        onto the end of the output dims list, which starts empty. The codes
 *        0  Copy     Pop one input dim and push it onto the output dims
 *        -1  Infer    Push a dim that is inferred later from all other output
 *        -2  CopyAll  Pop all remaining input dims and push them onto output
 *        -3  Merge2   Pop two input dims, multiply them, and push result
 *        -4  Split2   Pop one input dim, and read two next target shape specs,
 *        push them both onto output dims (either can be -1 and will
 *        be inferred from the other
 *        The exact mathematical behavior of these codes is given in the
 *        description of the 'shape' parameter. All non-codes (positive
 *        integers) just pop a dim off the input dims (if any), throw it away,
 *        Examples:
 *        Type     Input      Target            Output
 *        Copy     (2,3,4)    (4,0,2)           (4,3,2)
 *        Copy     (2,3,4)    (2,0,0)           (2,3,4)
 *        Infer    (2,3,4)    (6,1,-1)          (6,1,4)
 *        Infer    (2,3,4)    (3,-1,8)          (3,1,8)
 *        CopyAll  (9,8,7)    (-2)              (9,8,7)
 *        CopyAll  (9,8,7)    (9,-2)            (9,8,7)
 *        CopyAll  (9,8,7)    (-2,1,1)          (9,8,7,1,1)
 *        Merge2   (3,4)      (-3)              (12)
 *        Merge2   (3,4,5)    (-3,0)            (12,5)
 *        Merge2   (3,4,5)    (0,-3)            (3,20)
 *        Merge2   (3,4,5,6)  (-3,0,0)          (12,5,6)
 *        Merge2   (3,4,5,6)  (-3,-2)           (12,5,6)
 *        Split2   (12)       (-4,6,2)          (6,2)
 *        Split2   (12)       (-4,2,6)          (2,6)
 *        Split2   (12)       (-4,-1,6)         (2,6)
 *        Split2   (12,9)     (-4,2,6,0)        (2,6,9)
 *        Split2   (12,9,9,9) (-4,2,6,-2)       (2,6,9,9,9)
 *        Split2   (12,12)    (-4,2,-1,-4,-1,2) (2,6,6,2)
 *
 *
 *        From:src/operator/tensor/matrix_op.cc:61
 * \param data Input data to reshape.
 * \param target_shape (Deprecated! Use shape instead.) Target new shape. One
 *        and only one dim can be 0, in which case it will be inferred from
 * \param keep_highest (Deprecated! Use shape instead.) Whether keep the
 *        highest dim unchanged.If set to true, then the first dim in
 * \param shape Target shape, a tuple, t=(t_1,t_2,..,t_m).
 *        Let the input dims be s=(s_1,s_2,..,s_n).
 *        The output dims u=(u_1,u_2,..,u_p) are computed from s and t.
 *        The target shape tuple elements t_i are read in order, and used to
 *        t_i:       meaning:      behavior:
 *        +ve        explicit      u_p = t_i
 *        0          copy          u_p = s_i
 *        -1         infer         u_p = (Prod s_i) / (Prod u_j | j != p)
 *        -2         copy all      u_p = s_i, u_p+1 = s_i+1, ...
 *        -3         merge two     u_p = s_i * s_i+1
 *        -4,a,b     split two     u_p = a, u_p+1 = b | a * b = s_i
 *        The split directive (-4) in the target shape tuple is followed by two
 *        dimensions, one of which can be -1, which means it will be inferred
 *        The can only be one globally inferred dimension (-1), aside from any
 * \param reverse Whether to match the shapes from the backward. If reverse is
 *        true, 0 values in the `shape` argument will be searched from the
 *        backward. E.g the original shape is (10, 5, 4) and the shape
 *        argument is (-1, 0). If reverse is true, the new shape should be
 * \return new symbol
 */
inline Symbol Reshape(Symbol data,
                      Shape target_shape = Shape(0,0),
                      bool keep_highest = false,
                      Shape shape = Shape(),
                      bool reverse = false) {
  return Operator("Reshape")
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Flatten input into 2D by collapsing all the higher dimensions.
 *        A (d1, d2, ..., dK) tensor is flatten to (d1, d2* ... *dK) matrix.
 * \param data Input data to reshape.
 * \return new symbol
 */
inline Symbol Flatten(Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Transpose the input tensor and return a new one
 *
 *        From:src/operator/tensor/matrix_op.cc:93
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Expand the shape of array by inserting a new axis.
 *
 *        From:src/operator/tensor/matrix_op.cc:121
 * \param data Source input
 * \param axis Position (amongst axes) where new axis is to be inserted.
 * \return new symbol
 */
inline Symbol expand_dims(Symbol data,
                          int axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif (Crop the input tensor and return a new one.
 *
 *        Requirements
 *        ------------
 *        - the input and output (if explicitly given) are of the same data
 *        and on the same device.
 *        )
 *
 *        From:src/operator/tensor/matrix_op.cc:142
 * \param data Source input
 * \param begin starting coordinates
 * \param end ending coordinates
 * \return new symbol
 */
inline Symbol crop(Symbol data,
                   Shape begin,
                   Shape end) {
  return Operator("crop")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Slice the input along certain axis and return a sliced array.
 *
 *        From:src/operator/tensor/matrix_op.cc:197
 * \param data Source input
 * \param axis The axis to be sliced
 * \param begin The beginning index to be sliced
 * \param end The end index to be sliced
 * \return new symbol
 */
inline Symbol slice_axis(Symbol data,
                         int axis,
                         int begin,
                         int end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Flip the input tensor along axis and return a new one.
 *
 *        From:src/operator/tensor/matrix_op.cc:216
 * \param data Source input
 * \param axis The dimension to flip
 * \return new symbol
 */
inline Symbol flip(Symbol data,
                   int axis) {
  return Operator("flip")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate dot product of two matrices or two vectors.
 *
 *        From:src/operator/tensor/matrix_op.cc:228
 * \param lhs Left input
 * \param rhs Right input
 * \param transpose_a True if the first matrix is transposed.
 * \param transpose_b True if the second matrix is tranposed.
 * \return new symbol
 */
inline Symbol dot(Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false) {
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Calculate batched dot product of two matrices. (batch, M, K)
 *
 *        From:src/operator/tensor/matrix_op.cc:254
 * \param lhs Left input
 * \param rhs Right input
 * \param axis The dimension to flip
 * \return new symbol
 */
inline Symbol batch_dot(Symbol lhs,
                        Symbol rhs,
                        int axis) {
  return Operator("batch_dot")
           .SetParam("axis", axis)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_power(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_maximum(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_minimum(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_hypot(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_hypot")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Compute argmax
 *
 *        From:src/operator/tensor/broadcast_reduce_op_index.cc:11
 * \param data Source input
 * \param axis Empty or unsigned. The axis to perform the reduction.If left
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol argmax(Symbol data,
                     int axis = -1,
                     bool keepdims = false) {
  return Operator("argmax")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute argmin
 *
 *        From:src/operator/tensor/broadcast_reduce_op_index.cc:15
 * \param data Source input
 * \param axis Empty or unsigned. The axis to perform the reduction.If left
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol argmin(Symbol data,
                     int axis = -1,
                     bool keepdims = false) {
  return Operator("argmin")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif
 * \param src Source input
 * \return new symbol
 */
inline Symbol argmax_channel(Symbol src) {
  return Operator("argmax_channel")
           .SetInput("src", src)
           .CreateSymbol();
}

/*!
 * \breif Sum src along axis. If axis is empty, global reduction is performed
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:17
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol sum(Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("sum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute product of src along axis. If axis is empty, global reduction
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:27
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol prod(Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false) {
  return Operator("prod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Sum src along axis, ignoring NaN values. If axis is empty, global
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:37
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol nansum(Symbol data,
                     Shape axis = Shape(),
                     bool keepdims = false) {
  return Operator("nansum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute product of src along axis, ignoring NaN values. If axis is
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:47
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol nanprod(Symbol data,
                      Shape axis = Shape(),
                      bool keepdims = false) {
  return Operator("nanprod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute max along axis. If axis is empty, global reduction is
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:57
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol max(Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("max")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute min along axis. If axis is empty, global reduction is
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:67
 * \param data Source input
 * \param axis Empty or unsigned or tuple. The axes to perform the reduction.If
 * \param keepdims If true, the axis which is reduced is left in the result as
 * \return new symbol
 */
inline Symbol min(Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("min")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Broadcast src along axis
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:76
 * \param data Source input
 * \param axis The axes to perform the broadcasting.
 * \param size Target sizes of the broadcasting axes.
 * \return new symbol
 */
inline Symbol broadcast_axis(Symbol data,
                             Shape axis = Shape(),
                             Shape size = Shape()) {
  return Operator("broadcast_axis")
           .SetParam("axis", axis)
           .SetParam("size", size)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Broadcast src to shape
 *
 *        From:src/operator/tensor/broadcast_reduce_op_value.cc:83
 * \param data Source input
 * \param shape The shape of the desired array. We can set the dim to zero if
 *        it's same as the original. E.g `A = broadcast_to(B, shape=(10, 0,
 *        0))` has the same meaning as `A = broadcast_axis(B, axis=0,
 * \return new symbol
 */
inline Symbol broadcast_to(Symbol data,
                           Shape shape = Shape()) {
  return Operator("broadcast_to")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif
 * \param src Source input
 * \return new symbol
 */
inline Symbol norm(Symbol src) {
  return Operator("norm")
           .SetInput("src", src)
           .CreateSymbol();
}

/*!
 * \breif Negate src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:52
 * \param data Source input
 * \return new symbol
 */
inline Symbol negative(Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take absolute value of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:58
 * \param data Source input
 * \return new symbol
 */
inline Symbol abs(Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take sign of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:67
 * \param data Source input
 * \return new symbol
 */
inline Symbol sign(Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take round of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:76
 * \param data Source input
 * \return new symbol
 */
inline Symbol round(Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take ceil of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:81
 * \param data Source input
 * \return new symbol
 */
inline Symbol ceil(Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take floor of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:86
 * \param data Source input
 * \return new symbol
 */
inline Symbol floor(Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take round of the src to nearest integer
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:91
 * \param data Source input
 * \return new symbol
 */
inline Symbol rint(Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take round of the src to integer nearest 0
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:96
 * \param data Source input
 * \return new symbol
 */
inline Symbol fix(Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take square of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:101
 * \param data Source input
 * \return new symbol
 */
inline Symbol square(Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take square root of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:110
 * \param data Source input
 * \return new symbol
 */
inline Symbol sqrt(Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take reciprocal square root of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:119
 * \param data Source input
 * \return new symbol
 */
inline Symbol rsqrt(Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take exp of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:129
 * \param data Source input
 * \return new symbol
 */
inline Symbol exp(Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take log of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:135
 * \param data Source input
 * \return new symbol
 */
inline Symbol log(Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take base-10 log of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:141
 * \param data Source input
 * \return new symbol
 */
inline Symbol log10(Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take base-2 log of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:147
 * \param data Source input
 * \return new symbol
 */
inline Symbol log2(Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take sin of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:156
 * \param data Source input
 * \return new symbol
 */
inline Symbol sin(Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take `log(1 + x)` in a numerically stable way
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:165
 * \param data Source input
 * \return new symbol
 */
inline Symbol log1p(Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take `exp(x) - 1` in a numerically stable way
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:174
 * \param data Source input
 * \return new symbol
 */
inline Symbol expm1(Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take cos of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:183
 * \param data Source input
 * \return new symbol
 */
inline Symbol cos(Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take tan of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:192
 * \param data Source input
 * \return new symbol
 */
inline Symbol tan(Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take arcsin of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:201
 * \param data Source input
 * \return new symbol
 */
inline Symbol arcsin(Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take arccos of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:210
 * \param data Source input
 * \return new symbol
 */
inline Symbol arccos(Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take arctan of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:219
 * \param data Source input
 * \return new symbol
 */
inline Symbol arctan(Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take degrees of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:228
 * \param data Source input
 * \return new symbol
 */
inline Symbol degrees(Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take radians of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:237
 * \param data Source input
 * \return new symbol
 */
inline Symbol radians(Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take sinh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:246
 * \param data Source input
 * \return new symbol
 */
inline Symbol sinh(Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take cosh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:255
 * \param data Source input
 * \return new symbol
 */
inline Symbol cosh(Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take tanh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:264
 * \param data Source input
 * \return new symbol
 */
inline Symbol tanh(Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take arcsinh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:273
 * \param data Source input
 * \return new symbol
 */
inline Symbol arcsinh(Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take arccosh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:282
 * \param data Source input
 * \return new symbol
 */
inline Symbol arccosh(Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take arctanh of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:291
 * \param data Source input
 * \return new symbol
 */
inline Symbol arctanh(Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take the gamma function (extension of the factorial function) of the
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:300
 * \param data Source input
 * \return new symbol
 */
inline Symbol gamma(Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Take gammaln (log of the absolute value of gamma(x)) of the src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:309
 * \param data Source input
 * \return new symbol
 */
inline Symbol gammaln(Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate Smooth L1 Loss(lhs, scalar)
 *
 *        From:src/operator/tensor/elemwise_binary_scalar_op_extended.cc:63
 * \param data source input
 * \param scalar scalar input
 * \return new symbol
 */
inline Symbol smooth_l1(Symbol data,
                        mx_float scalar) {
  return Operator("smooth_l1")
           .SetParam("scalar", scalar)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Map integer index to vector representations (embeddings). Those
 *        embeddings are learnable parameters. For a input of shape (d1, ...,
 *        dK), the output shape is (d1, ..., dK, output_dim). All the input
 *
 *        From:src/operator/tensor/indexing_op.cc:17
 * \param data Input data to the EmbeddingOp.
 * \param weight Embedding weight matrix.
 * \param input_dim vocabulary size of the input indices.
 * \param output_dim dimension of the embedding vectors.
 * \return new symbol
 */
inline Symbol Embedding(Symbol data,
                        Symbol weight,
                        int input_dim,
                        int output_dim) {
  return Operator("Embedding")
           .SetParam("input_dim", input_dim)
           .SetParam("output_dim", output_dim)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_equal(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_not_equal(Symbol lhs,
                                  Symbol rhs) {
  return Operator("broadcast_not_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_greater(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_greater")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_greater_equal(Symbol lhs,
                                      Symbol rhs) {
  return Operator("broadcast_greater_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_lesser(Symbol lhs,
                               Symbol rhs) {
  return Operator("broadcast_lesser")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_lesser_equal(Symbol lhs,
                                     Symbol rhs) {
  return Operator("broadcast_lesser_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Sample a uniform distribution
 * \param low The lower bound of distribution
 * \param high The upper bound of distribution
 * \param shape The shape of the output
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n).Only used
 * \return new symbol
 */
inline Symbol uniform(mx_float low = 0,
                      mx_float high = 1,
                      Shape shape = Shape(),
                      const std::string& ctx = "") {
  return Operator("uniform")
           .SetParam("low", low)
           .SetParam("high", high)
           .SetParam("shape", shape)
           .CreateSymbol();
}

/*!
 * \breif Sample a normal distribution
 * \param loc Mean of the distribution.
 * \param scale Standard deviation of the distribution.
 * \param shape The shape of the output
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n).Only used
 * \return new symbol
 */
inline Symbol normal(mx_float loc = 0,
                     mx_float scale = 1,
                     Shape shape = Shape(),
                     const std::string& ctx = "") {
  return Operator("normal")
           .SetParam("loc", loc)
           .SetParam("scale", scale)
           .SetParam("shape", shape)
           .CreateSymbol();
}

/*!
 * \breif Perform element sum of inputs
 *
 *        From:src/operator/tensor/elemwise_sum.cc:56
 * \param args List of input tensors
 * \return new symbol
 */
inline Symbol ElementWiseSum(const std::vector<Symbol>& args) {
  return Operator("ElementWiseSum")
(args)
           .CreateSymbol();
}

/*!
 * \breif Updater function for sgd optimizer
 * \return new symbol
 */
inline Symbol sgd_update() {
  return Operator("sgd_update")
           .CreateSymbol();
}

/*!
 * \breif Updater function for sgd optimizer
 * \return new symbol
 */
inline Symbol sgd_mom_update() {
  return Operator("sgd_mom_update")
           .CreateSymbol();
}

/*!
 * \breif Updater function for adam optimizer
 * \return new symbol
 */
inline Symbol adam_update() {
  return Operator("adam_update")
           .CreateSymbol();
}

/*!
 * \breif Calculate cross_entropy(lhs, one_hot(rhs))
 *
 *        From:src/operator/loss_binary_op.cc:12
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif An operator taking in a n-dimensional input tensor (n > 2), and
 *        normalizing the input by subtracting the mean and variance
 *        calculated over the spatial dimensions. This is an implemention of
 *        the operator described in "Instance Normalization: The Missing
 *        Ingredient for Fast Stylization", D. Ulyanov, A. Vedaldi, V.
 *        Lempitsky, 2016 (arXiv:1607.08022v2). This layer is similar to batch
 *        normalization, with two differences: first, the normalization is
 *        carried out per example ('instance'), not over a batch. Second, the
 *        same normalization is applied both at test and train time. This
 * \param data A n-dimensional tensor (n > 2) of the form [batch, channel,
 * \param gamma A vector of length 'channel', which multiplies the normalized
 * \param beta A vector of length 'channel', which is added to the product of
 * \param eps Epsilon to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \breif Support Vector Machine based transformation on input, backprop L2-SVM
 * \param data Input data to svm.
 * \param label Label data.
 * \param margin Scale the DType(param_.margin) for activation size
 * \param regularization_coefficient Scale the coefficient responsible for
 * \param use_linear If set true, uses L1-SVM objective function. Default uses
 * \return new symbol
 */
inline Symbol SVMOutput(Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Apply a recurrent layer to input.
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p Dropout probability, fraction of the input that gets dropped out at
 * \param state_outputs Whether to have the states as symbol outputs.
 * \return new symbol
 */
inline Symbol RNN(Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  int state_size,
                  int num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .CreateSymbol();
}

/*!
 * \breif Cast array to a different data type.
 * \param data Input data to cast function.
 * \param dtype Target data type.
 * \return new symbol
 */
inline Symbol Cast(Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Crop the 2nd and 3rd dim of input data, with the corresponding size
 *        of h_w or with width and height of the second input symbol, i.e.,
 *        with one input, we need h_w to specify the crop height and width,
 * \param data Tensor or List of Tensors, the second input will be used as
 * \param num_args Number of inputs for crop, if equals one, then we will use
 *        the h_wfor crop height and width, else if equals two, then we will
 *        use the heightand width of the second input symbol, we name
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and weight: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it
 * \return new symbol
 */
inline Symbol Crop(Symbol data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Reverses the elements of each sequence. Takes an n-dimensional tensor
 *        of the form [max sequence length, batchsize, other dims] and returns
 *        a tensor of the same shape. This operator takes an optional input
 *        tensor sequence_length of positive ints of dimension [batchsize]
 *        when the sequence_length option is set to true. This allows the
 *        operator to handle variable-length sequences. If sequence_length is
 *        false, then each example in the batch is assumed to have the max
 * \param data n-dimensional input tensor of the form [max sequence length,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \return new symbol
 */
inline Symbol SequenceReverse(Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Apply spatial transformer to input feature map.
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type
 *        is affine, and the name of loc symbol should better starts with
 *        'stn_loc', so that initialization it with iddentify tranform, or you
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \return new symbol
 */
inline Symbol SpatialTransformer(Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0)) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol();
}

/*!
 * \breif Apply swapaxis to input.
 * \param data Input data to the SwapAxisOp.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(Symbol data,
                       int dim1 = 0,
                       int dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Pads an n-dimensional input tensor. Allows for precise control of the
 *        padding type and how much padding to apply on both sides of a given
 * \param data An n-dimensional input tensor.
 * \param mode Padding type to use. "constant" pads all values with a constant
 *        value, the value of which can be specified with the constant_value
 * \param pad_width A tuple of padding widths of length 2*r, where r is the
 *        rank of the input tensor, specifying number of values padded to the
 *        edges of each axis. (before_1, after_1, ... , before_N, after_N)
 *        unique pad widths for each axis. Equivalent to pad_width in
 * \param constant_value This option is only used when mode is "constant". This
 *        value will be used as the padding value. Defaults to 0 if not
 * \return new symbol
 */
inline Symbol Pad(Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Perform a softmax transformation on input, backprop with logloss.
 * \param data Input data to softmax.
 * \param label Label data, can also be probability value with same shape as
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the label value will be ignored during backward (only
 * \param multi_output If set to true, for a (n,k,x_1,..,x_n) dimensional input
 *        tensor, softmax will generate n*x_1*...*x_n output, each has k
 * \param use_ignore If set to true, the ignore_label value will not contribute
 * \param preserve_shape If true, for a (n_1, n_2, ..., n_d, k) dimensional
 *        input tensor, softmax will generate (n1, n2, ..., n_d, k) output,
 * \param normalization If set to null, op will do nothing on output
 *        gradient.If set to batch, op will normalize gradient by divide batch
 *        sizeIf set to valid, op will normalize gradient by divide sample not
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol SoftmaxOutput(Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::null,
                            bool out_grad = false) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif DEPRECATED: Perform a softmax transformation on input. Please use
 * \param data Input data to softmax.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the label value will be ignored during backward (only
 * \param multi_output If set to true, for a (n,k,x_1,..,x_n) dimensional input
 *        tensor, softmax will generate n*x_1*...*x_n output, each has k
 * \param use_ignore If set to true, the ignore_label value will not contribute
 * \param preserve_shape If true, for a (n_1, n_2, ..., n_d, k) dimensional
 *        input tensor, softmax will generate (n1, n2, ..., n_d, k) output,
 * \param normalization If set to null, op will do nothing on output
 *        gradient.If set to batch, op will normalize gradient by divide batch
 *        sizeIf set to valid, op will normalize gradient by divide sample not
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol Softmax(Symbol data,
                      mx_float grad_scale = 1,
                      mx_float ignore_label = -1,
                      bool multi_output = false,
                      bool use_ignore = false,
                      bool preserve_shape = false,
                      SoftmaxNormalization normalization = SoftmaxNormalization::null,
                      bool out_grad = false) {
  static const char *SoftmaxNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("Softmax")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply activation function to input.
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(Symbol data,
                        LeakyReLUActType act_type = LeakyReLUActType::leaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.334) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "leaky",
    "prelu",
    "rrelu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Takes the last element of a sequence. Takes an n-dimensional tensor
 *        of the form [max sequence length, batchsize, other dims] and returns
 *        a (n-1)-dimensional tensor of the form [batchsize, other dims]. This
 *        operator takes an optional input tensor sequence_length of positive
 *        ints of dimension [batchsize] when the sequence_length option is set
 *        to true. This allows the operator to handle variable-length
 *        sequences. If sequence_length is false, then each example in the
 * \param data n-dimensional input tensor of the form [max sequence length,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \return new symbol
 */
inline Symbol SequenceLast(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Set the l2 norm of each instance to a constant.
 * \param data Input data to the L2NormalizationOp.
 * \param eps Epsilon to prevent div 0
 * \param mode Normalization Mode. If set to instance, this operator will
 *        compute a norm for each instance in the batch; this is the default
 *        mode. If set to channel, this operator will compute a cross channel
 *        norm at each position of each instance. If set to spatial, this
 * \return new symbol
 */
inline Symbol L2Normalization(Symbol data,
                              mx_float eps = 1e-10,
                              L2NormalizationMode mode = L2NormalizationMode::instance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply convolution to input then add a bias.
 * \param data Input data to the ConvolutionOp.
 * \param nsize normalization window width in elements.
 * \param alpha value of the alpha variance scaling parameter in the
 * \param beta value of the beta power parameter in the normalization formula
 * \param knorm value of the k parameter in normalization formula
 * \return new symbol
 */
inline Symbol LRN(Symbol data,
                  int nsize,
                  mx_float alpha = 0.0001,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply correlation to inputs
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(Symbol data1,
                          Symbol data2,
                          int kernel_size = 1,
                          int max_displacement = 1,
                          int stride1 = 1,
                          int stride2 = 1,
                          int pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol();
}

/*!
 * \breif Sets all elements outside the sequence to a constant value. Takes an
 *        n-dimensional tensor of the form [max sequence length, batchsize,
 *        other dims] and returns a tensor of the same shape. This operator
 *        takes an optional input tensor sequence_length of positive ints of
 *        dimension [batchsize] when the sequence_length option is set to
 *        true. This allows the operator to handle variable-length sequences.
 *        If sequence_length is false, then each example in the batch is
 *        assumed to have the max sequence length, and this operator becomes
 * \param data n-dimensional input tensor of the form [max sequence length,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \param value The value to be used as a mask.
 * \return new symbol
 */
inline Symbol SequenceMask(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Get output from a symbol and pass 0 gradient back
 * \param data Input data.
 * \return new symbol
 */
inline Symbol BlockGrad(Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply a sparse regularization to the output a sigmoid activation
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(Symbol data,
                                        mx_float sparseness_target = 0.1,
                                        mx_float penalty = 0.001,
                                        mx_float momentum = 0.9) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Perform nearest neighboor/bilinear up sampling to inputs
 * \param data Array of tensors to upsample
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor
 *        upsampling, this can be 1-N; the size of output will
 *        be(scale*h_0,scale*w_0) and all other inputs will be upsampled to
 *        thesame size. For bilinear upsampling this must be 2; 1 input and 1
 * \param num_filter Input filter. Only used by bilinear sample_type.
 * \param multi_input_mode How to handle multiple input. concat means
 *        concatenate upsampled images along the channel dimension. sum means
 *        add all images together, only available for nearest neighbor
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::vector<Symbol>& data,
                         int scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         int num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::concat,
                         int64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol();
}

/*!
 * \breif Apply deconvolution to input then add a bias.
 * \param data Input data to the DeconvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel deconvolution kernel size: (y, x)
 * \param num_filter deconvolution filter(channel) number
 * \param stride deconvolution stride: (y, x)
 * \param pad pad for deconvolution: (y, x), a good number is : (kernel-1)/2,
 *        if target_shape set, pad will be ignored and will be computed
 * \param adj adjustment for output shape: (y, x), if target_shape set, adj
 * \param target_shape output shape with targe shape : (y, x)
 * \param num_group number of groups partition
 * \param workspace Tmp workspace for deconvolution (MB)
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol Deconvolution(Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            int num_filter,
                            Shape stride = Shape(1,1),
                            Shape pad = Shape(0,0),
                            Shape adj = Shape(0,0),
                            Shape target_shape = Shape(0,0),
                            int num_group = 1,
                            int64_t workspace = 512,
                            bool no_bias = true) {
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Apply convolution to input then add a bias.
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions. Equivalent to slicing input
 *        partitions, apply convolution on each, then concatenate the results
 * \param workspace Maximum tmp workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution(Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          int num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          int num_group = 1,
                          int64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::None,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::None) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Performs region-of-interest pooling on inputs. Resize bounding box
 *        coordinates by spatial_scale and crop input feature maps
 *        accordingly. The cropped feature maps are pooled by max pooling to a
 *        fixed size output indicated by pooled_size. batch_size will change
 * \param data Input data to the pooling operator, a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1,
 *        x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners
 *        of designated region of interest. batch_index indicates the index of
 * \param pooled_size fix pooled size: (h, w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image
 *        height (or w). Equals the reciprocal of total stride in
 * \return new symbol
 */
inline Symbol ROIPooling(Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol();
}

/*!
 * \breif Apply batch normalization to input.
 * \param data Input data to batch normalization
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of
 *        local batch-norm. This will force change batch-norm into a scale
 * \return new symbol
 */
inline Symbol CuDNNBatchNorm(Symbol data,
                             mx_float eps = 0.001,
                             mx_float momentum = 0.9,
                             bool fix_gamma = true,
                             bool use_global_stats = false) {
  return Operator("CuDNNBatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply matrix multiplication to input then add a bias.
 *        It maps the input of shape `(batch_size, input_dim)` to the shape of
 *        `(batch_size, num_hidden)`. Learnable parameters include the weights
 *        of the linear transform and an optional bias vector.
 * \param data Input data to the FullyConnectedOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol FullyConnected(Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Use linear regression for final output, this is used on final output
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Use mean absolute error regression for final output, this is used on
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Use Logistic regression for final output, this is used on final
 *        Logistic regression is suitable for binary classification or
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Get output from a symbol and pass 1 gradient back. This is used as a
 *        terminal loss if unary and binary operator are used to composite a
 * \param data Input data.
 * \param grad_scale gradient scale as a supplement to unary and binary
 * \param valid_thresh regard element valid when x > valid_thresh, this is used
 * \param normalization If set to null, op will not normalize on output
 *        gradient.If set to batch, op will normalize gradient by divide batch
 *        size.If set to valid, op will normalize gradient by divide # sample
 * \return new symbol
 */
inline Symbol MakeLoss(Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::null) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Slice input equally along specified axis
 * \param num_outputs Number of outputs to be sliced.
 * \param axis Dimension along which to slice.
 * \param squeeze_axis If true AND the sliced dimension becomes 1, squeeze that
 * \return new symbol
 */
inline Symbol SliceChannel(int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .CreateSymbol();
}

/*!
 * \breif Perform a feature concat on channel dim (defaut is 1) over all
 * \param data List of tensors to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol();
}

/*!
 * \breif Elementwise activation function.
 *
 *        The following activation types are supported (operations are applied
 *        scalar of the input tensor):
 *
 *        - `relu`: Rectified Linear Unit, `y = max(x, 0)`
 *        - `sigmoid`: `y = 1 / (1 + exp(-x))`
 *        - `tanh`: Hyperbolic tangent, `y = (exp(x) - exp(-x)) / (exp(x) +
 *        - `softrelu`: Soft ReLU, or SoftPlus, `y = log(1 + exp(x))`
 *
 *        See `LeakyReLU` for other activations with parameters.
 *
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Custom operator implemented in frontend.
 * \param op_type Type of custom operator. Must be registered first.
 * \return new symbol
 */
inline Symbol Custom(const std::string& op_type) {
  return Operator("Custom")
           .CreateSymbol();
}

/*!
 * \breif Perform spatial pooling on inputs.
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current
 * \param pooling_convention Pooling convention to be applied.kValid is default
 *        setting of Mxnet and rounds down the output pooling size.kFull is
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling(Symbol data,
                      Shape kernel,
                      PoolingPoolType pool_type,
                      bool global_pool = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::valid,
                      Shape stride = Shape(1,1),
                      Shape pad = Shape(0,0)) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply batch normalization to input.
 * \param data Input data to batch normalization
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of
 *        local batch-norm. This will force change batch-norm into a scale
 * \return new symbol
 */
inline Symbol BatchNorm(Symbol data,
                        mx_float eps = 0.001,
                        mx_float momentum = 0.9,
                        bool fix_gamma = true,
                        bool use_global_stats = false) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply dropout to input.
 *        During training, each element of the input is randomly set to zero
 *        And then the whole tensor is rescaled by 1/(1-p) to keep the
 *        before applying dropout. During the test time, this behaves as an
 *
 * \param data Input data to dropout.
 * \param p Fraction of the input that gets dropped out at training time
 * \return new symbol
 */
inline Symbol Dropout(Symbol data,
                      mx_float p = 0.5) {
  return Operator("Dropout")
           .SetParam("p", p)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply softmax activation to input. This is intended for internal
 *        layers. For output (loss layer) please use SoftmaxOutput. If
 *        mode=instance, this operator will compute a softmax for each
 *        instance in the batch; this is the default mode. If mode=channel,
 *        this operator will compute a num_channel-class softmax at each
 *        position of each instance; this can be used for fully convolutional
 * \param data Input data to activation function.
 * \param mode Softmax Mode. If set to instance, this operator will compute a
 *        softmax for each instance in the batch; this is the default mode. If
 *        set to channel, this operator will compute a num_channel-class
 *        softmax at each position of each instance; this can be used for
 * \return new symbol
 */
inline Symbol SoftmaxActivation(Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::instance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Choose one element from each line(row for python, column for R/Julia)
 *        in lhs according to index indicated by rhs. This function assume rhs
 * \param lhs Left operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol choose_element_0index(Symbol lhs,
                                    Symbol rhs) {
  return Operator("choose_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Fill one element of each line(row for python, column for R/Julia) in
 *        lhs according to index indicated by rhs and values indicated by mhs.
 * \param lhs Left operand to the function.
 * \param mhs Middle operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol fill_element_0index(Symbol lhs,
                                  Symbol mhs,
                                  Symbol rhs) {
  return Operator("fill_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("mhs", mhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Clip ndarray elements to range (a_min, a_max)
 * \param src Source input
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(Symbol src,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("src", src)
           .CreateSymbol();
}

} //namespace cpp
} //namespace mxnet
#endif //ifndef _MXNETOP_H
