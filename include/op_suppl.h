/*!
*  Copyright (c) 2016 by Contributors
* \file op_suppl.h
* \brief A supplement and amendment of the operators from op.h
* \author Zhang Chen, zhubuntu
*/

#ifndef OP_SUPPL_H
#define OP_SUPPL_H

#include <string>
#include <vector>
#include "base.h"
#include "shape.h"
#include "operator.h"
#include "MxNetCpp.h"

namespace mxnet {
namespace cpp {

inline Symbol _Plus(Symbol lhs, Symbol rhs) {
  return Operator("_Plus")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
inline Symbol _Mul(Symbol lhs, Symbol rhs) {
  return Operator("_Mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
inline Symbol _Minus(Symbol lhs, Symbol rhs) {
  return Operator("_Minus")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
inline Symbol _Div(Symbol lhs, Symbol rhs) {
  return Operator("_Div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
inline Symbol _Power(Symbol lhs, Symbol rhs) {
  return Operator("_Power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
inline Symbol _Maximum(Symbol lhs, Symbol rhs) {
  return Operator("_Maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
inline Symbol _Minimum(Symbol lhs, Symbol rhs) {
  return Operator("_Minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
inline Symbol _PlusScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_PlusScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
inline Symbol _MinusScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_MinusScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
inline Symbol _MulScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_MulScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
inline Symbol _DivScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_DivScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
inline Symbol _PowerScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_PowerScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
inline Symbol _MaximumScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_MaximumScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
inline Symbol _MinimumScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_MinimumScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
// TODO(zhangcheng-qinyinghua)
//  make crop function run in op.h
//  This function is due to [zhubuntu](https://github.com/zhubuntu)
inline Symbol Crop(const std::string& symbol_name,
    int num_args,
    Symbol data,
    Symbol crop_like,
    Shape offset = Shape(0, 0),
    Shape h_w = Shape(0, 0),
    bool center_crop = false) {
  return Operator("Crop")
    .SetParam("num_args", num_args)
    .SetParam("offset", offset)
    .SetParam("h_w", h_w)
    .SetParam("center_crop", center_crop)
    .SetInput("arg0", data)
    .SetInput("arg1", crop_like)
    .CreateSymbol(symbol_name);
}


/*!
 * \breif Slice input equally along specified axis.
 * \param symbol_name name of the resulting symbol.
 * \param data input symbol. 
 * \param num_outputs Number of outputs to be sliced. 
 * \param axis Dimension along which to slice. 
 * \param squeeze_axis If true AND the sliced dimension becomes 1, squeeze that dimension. 
 * \return new symbol
 */
inline Symbol SliceChannel(const std::string& symbol_name,
                           Symbol data,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
      .SetParam("num_outputs", num_outputs)
      .SetParam("axis", axis)
      .SetParam("squeeze_axis", squeeze_axis) (data)
      .CreateSymbol(symbol_name);
}

inline Symbol ConvolutionNoBias(const std::string& symbol_name,
                                Symbol data,
                                Symbol weight,
                                Symbol bias,
                                Shape kernel,
                                int num_filter,
                                Shape stride = Shape(1,1),
                                Shape dilate = Shape(1,1),
                                Shape pad = Shape(0,0),
                                int num_group = 1,
                                int64_t workspace = 512) {
  return Operator("Convolution")
      .SetParam("kernel", kernel)
      .SetParam("num_filter", num_filter)
      .SetParam("stride", stride)
      .SetParam("dilate", dilate)
      .SetParam("pad", pad)
      .SetParam("num_group", num_group)
      .SetParam("workspace", workspace)
      .SetParam("no_bias", true)
      .SetInput("data", data)
      .SetInput("weight", weight)
      .CreateSymbol(symbol_name);
}

}  // namespace cpp
}  // namespace mxnet


#endif /* end of include guard: OP_SUPPL_H */

