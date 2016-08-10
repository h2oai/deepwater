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

/*! \breif Activation function to be applied. 
*/
enum class ActivationActType {
  relu = 0,
  sigmoid = 1,
  softrelu = 2,
  tanh = 3
};

/*!
 * \breif Apply activation function to input.
 *        Softmax Activation is only available with CUDNN on GPUand will be
 *        computed at each location across channel if input is 4D.
 * \param symbol_name name of the resulting symbol.
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

inline Symbol Activation(const std::string& symbol_name,
                         Symbol data,
                         const std::string& act_type) {
  return Operator("Activation")
      .SetParam("act_type", act_type.c_str())
      .SetInput("data", data)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply batch normalization to input. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to batch normalization.
 * \param eps Epsilon to prevent div 0.
 * \param momentum Momentum for moving average.
 * \param fix_gamma Fix gamma while training.
 * \param use_global_stats Whether use global moving statistics instead of
 *        local batch-norm. This will force change batch-norm into a scale
 *        shift operator.
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
 * \breif Apply batch normalization to input. 
 * \param data Input data to batch normalization.
 * \param eps Epsilon to prevent div 0.
 * \param momentum Momentum for moving average.
 * \param fix_gamma Fix gamma while training.
 * \param use_global_stats Whether use global moving statistics instead of
 *        local batch-norm. This will force change batch-norm into a scale
 *        shift operator.
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
 * \breif Apply batch normalization to input. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to batch normalization.
 * \param eps Epsilon to prevent div 0.
 * \param momentum Momentum for moving average.
 * \param fix_gamma Fix gamma while training.
 * \param use_global_stats Whether use global moving statistics instead of
 *        local batch-norm. This will force change batch-norm into a scale
 *        shift operator.
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
 * \breif Get output from a symbol and pass 0 gradient back.
 * \param symbol_name name of the resulting symbol.
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
 * \breif Take sum of the src.
 *        The result will be ndarray of shape (1,) on the same device.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol sum(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs) {
  return Operator("sum")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take sum on medium dimension of the 3D src. 
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol sum_mid_internal(const std::string& symbol_name,
                               Symbol lhs,
                               Symbol rhs) {
  return Operator("sum_mid_internal")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
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
 * \param symbol_name name of the resulting symbol.
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
 * \breif Perform an feature concat on channel dim (defaut is 1) over all.
 * \param symbol_name name of the resulting symbol.
 * \param data List of tensors to concatenate.
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

/*!
 * \breif Apply convolution to input then add a bias. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to the ConvolutionOp. 
 * \param weight Weight matrix. 
 * \param bias Bias parameter. 
 * \param kernel convolution kernel size: (y, x).
 * \param num_filter convolution filter(channel) number.
 * \param stride convolution stride: (y, x).
 * \param dilate convolution dilate: (y, x).
 * \param pad pad for convolution: (y, x).
 * \param num_group Number of groups partition.
 *        This option is not supported by CuDNN, you can use SliceChannel to
 *        num_group,apply convolution and concat instead to achieve the same need.
 * \param workspace Tmp workspace for convolution (MB). 
 * \param no_bias Whether to disable bias parameter. 
 * \return new symbol
 */
inline Symbol Convolution(const std::string& symbol_name,
                          Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          int num_filter,
                          Shape stride = Shape(1,1),
                          Shape dilate = Shape(1,1),
                          Shape pad = Shape(0,0),
                          int num_group = 1,
                          int64_t workspace = 512,
                          bool no_bias = false) {
  return Operator("Convolution")
      .SetParam("kernel", kernel)
      .SetParam("num_filter", num_filter)
      .SetParam("stride", stride)
      .SetParam("dilate", dilate)
      .SetParam("pad", pad)
      .SetParam("num_group", num_group)
      .SetParam("workspace", workspace)
      .SetParam("no_bias", no_bias)
      .SetInput("data", data)
      .SetInput("weight", weight)
      .SetInput("bias", bias)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Crop the 2nd and 3rd dim of input data, with the corresponding size
 *        of h_w or with width and height of the second input symbol, i.e.,
 *        with one input, we need h_w to specify the crop height and width,
 *        otherwise the second input symbol's size will be used
 * \param symbol_name name of the resulting symbol.
 * \param data Tensor or List of Tensors, the second input will be used as
 *        crop_like shape reference
 * \param num_args Number of inputs for crop, if equals one, then we will use
 *        the h_wfor crop height and width, else if equals two, then we will
 *        use the heightand width of the second input symbol, we name crop_like here
 * \param offset crop offset coordinate: (y, x).
 * \param h_w crop height and weight: (h, w).
 * \param center_crop If set to true, then it will use be the center_crop,or it
 *        will crop using the shape of crop_like
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
 * \breif Custom operator implemented in frontend. 
 * \param symbol_name name of the resulting symbol.
 * \param op_type Type of custom operator.
 *        Must be registered first.
 * \return new symbol
 */
inline Symbol Custom(const std::string& symbol_name,
                     const std::string& op_type) {
  return Operator("Custom")
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply deconvolution to input then add a bias. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to the DeconvolutionOp. 
 * \param weight Weight matrix. 
 * \param bias Bias parameter. 
 * \param kernel deconvolution kernel size: (y, x).
 * \param num_filter deconvolution filter(channel) number.
 * \param stride deconvolution stride: (y, x).
 * \param pad pad for deconvolution: (y, x).
 * \param num_group number of groups partition.
 * \param workspace Tmp workspace for deconvolution (MB).
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
                            int num_group = 1,
                            int64_t workspace = 512,
                            bool no_bias = true) {
  return Operator("Deconvolution")
      .SetParam("kernel", kernel)
      .SetParam("num_filter", num_filter)
      .SetParam("stride", stride)
      .SetParam("pad", pad)
      .SetParam("num_group", num_group)
      .SetParam("workspace", workspace)
      .SetParam("no_bias", no_bias)
      .SetInput("data", data)
      .SetInput("weight", weight)
      .SetInput("bias", bias)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply dropout to input.
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to dropout. 
 * \param p Fraction of the input that gets dropped out at training time.
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

/*!
 * \breif lhs add rhs with broadcast.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol broadcast_plus(const std::string& symbol_name,
                             Symbol lhs,
                             Symbol rhs) {
  return Operator("broadcast_plus")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif lhs minus rhs with broadcast.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol broadcast_minus(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_minus")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif lhs multiple rhs with broadcast.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
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
 * \breif lhs divide rhs with broadcast.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
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
 * \breif Perform an elementwise sum over all the inputs. 
 * \param symbol_name name of the resulting symbol.
 * \param num_args Number of inputs to be summed. 
 * \return new symbol
 */
inline Symbol ElementWiseSum(const std::string& symbol_name,
                             int num_args) {
  return Operator("ElementWiseSum")
      .SetParam("num_args", num_args)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take absolute value of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol abs(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs) {
  return Operator("abs")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take sign value of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol sign(const std::string& symbol_name,
                   Symbol lhs,
                   Symbol rhs) {
  return Operator("sign")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take round value of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol round(const std::string& symbol_name,
                    Symbol lhs,
                    Symbol rhs) {
  return Operator("round")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take ceil value of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol ceil(const std::string& symbol_name,
                   Symbol lhs,
                   Symbol rhs) {
  return Operator("ceil")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take floor value of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol floor(const std::string& symbol_name,
                    Symbol lhs,
                    Symbol rhs) {
  return Operator("floor")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take square of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol square(const std::string& symbol_name,
                     Symbol lhs,
                     Symbol rhs) {
  return Operator("square")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take sqrt of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol sqrt(const std::string& symbol_name,
                   Symbol lhs,
                   Symbol rhs) {
  return Operator("sqrt")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take rsqrt of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol rsqrt(const std::string& symbol_name,
                    Symbol lhs,
                    Symbol rhs) {
  return Operator("rsqrt")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take exp of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol exp(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs) {
  return Operator("exp")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take log of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol log(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs) {
  return Operator("log")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take cos of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol cos(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs) {
  return Operator("cos")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Take sin of the src.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol sin(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs) {
  return Operator("sin")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Get embedding for one-hot input.
 *        A n-dimensional input tensor will be trainsformed into a
 *        (n+1)-dimensional tensor, where a new dimension is added for the
 *        embedding results.
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to the EmbeddingOp. 
 * \param weight Enbedding weight matrix. 
 * \param input_dim input dim of one-hot encoding.
 * \param output_dim output dim of embedding.
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
 * \breif Apply matrix multiplication to input then add a bias. 
 * \param symbol_name name of the resulting symbol.
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
 * \breif Apply a sparse regularization to the output a sigmoid activation function. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data. 
 * \param sparseness_target The sparseness target.
 * \param penalty The tradeoff parameter for the sparseness penalty.
 * \param momentum The momentum for running average.
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

/*!
 * \breif Set the l2 norm of each instance to a constant. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to the L2NormalizationOp. 
 * \param eps Epsilon to prevent div 0.
 * \return new symbol
 */
inline Symbol L2Normalization(const std::string& symbol_name,
                              Symbol data,
                              mx_float eps = 1e-010) {
  return Operator("L2Normalization")
      .SetParam("eps", eps)
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
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to activation function. 
 * \param act_type Activation function to be applied. 
 * \param slope Init slope for the activation.
 *        (For leaky and elu only)
 * \param lower_bound Lower bound of random slope.
 *        (For rrelu only)
 * \param upper_bound Upper bound of random slope.
 *        (For rrelu only)
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
 * \breif Calculate cross_entropy(lhs, one_hot(rhs)).
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(const std::string& symbol_name,
                                    Symbol lhs,
                                    Symbol rhs) {
  return Operator("softmax_cross_entropy")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply convolution to input then add a bias. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to the ConvolutionOp. 
 * \param nsize normalization window width in elements. 
 * \param alpha value of the alpha variance scaling parameter in the
 *        normalization formula
 * \param beta value of the beta power parameter in the normalization formula.
 * \param knorm value of the k parameter in normalization formula.
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
 * \breif Get output from a symbol and pass 1 gradient back.
 *        This is used as a terminal loss if unary and binary operator are used
 *        to composite a loss with no declaration of backward dependency
 * \param symbol_name name of the resulting symbol.
 * \param data Input data. 
 * \param grad_scale gradient scale as a supplement to unary and binary operators.
 * \return new symbol
 */
inline Symbol MakeLoss(const std::string& symbol_name,
                       Symbol data,
                       mx_float grad_scale = 1) {
  return Operator("MakeLoss")
      .SetParam("grad_scale", grad_scale)
      .SetInput("data", data)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Transpose the input symbol and return a new one.
 * \param data the symbol to transpose.
 * \return new symbol
 */
inline Symbol transpose(Symbol data) {
  return Operator("transpose")
      .SetInput("data", data)
      .CreateSymbol();
}

/*!
 * \breif Calculate dot product of two matrices or two vectors.
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol dot(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs) {
  return Operator("dot")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*! \breif Pooling type to be applied. 
*/
enum class PoolingPoolType {
  avg = 0,
  max = 1,
  sum = 2
};

/*!
 * \breif Perform spatial pooling on inputs. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to the pooling operator. 
 * \param kernel pooling kernel size: (y, x).
 * \param pool_type Pooling type to be applied. 
 * \param global_pool Ignore kernel size, do global pooling based on current
 *        input feature map. This is useful for input with different shape
 * \param stride stride: for pooling (y, x).
 * \param pad pad for pooling: (y, x).
 * \return new symbol
 */
inline Symbol Pooling(const std::string& symbol_name,
                      Symbol data,
                      Shape kernel,
                      PoolingPoolType pool_type,
                      bool global_pool = false,
                      Shape stride = Shape(1,1),
                      Shape pad = Shape(0,0)) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  return Operator("Pooling")
      .SetParam("kernel", kernel)
      .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
      .SetParam("global_pool", global_pool)
      .SetParam("stride", stride)
      .SetParam("pad", pad)
      .SetInput("data", data)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Use linear regression for final output, this is used on final output of a net. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to function. 
 * \param label Input label to function. 
 * \param grad_scale Scale the gradient by a float factor.
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
 *        final output of a net.
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to function. 
 * \param label Input label to function. 
 * \param grad_scale Scale the gradient by a float factor.
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
 * \breif Use Logistic regression for final output, this is used on final output of a net.
 *        Logistic regression is suitable for binary classification or
 *        probability prediction tasks.
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to function. 
 * \param label Input label to function. 
 * \param grad_scale Scale the gradient by a float factor.
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

/*!
 * \breif Reshape input to target shape.
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to reshape. 
 * \param target_shape (Deprecated! Use shape instead.
 *        ) Target new shape. One and only one dim can be 0, in which case it
 *        will be inferred from the rest of dims
 * \param keep_highest (Deprecated! Use shape instead.
 *        ) Whether keep the highest dim unchanged.If set to yes, than the
 *        first dim in target_shape is ignored,and always fixed as input
 * \param shape Target new shape.
 *        If the dim is same, set it to 0. If the dim is set to be -1, it will
 *        be inferred from the rest of dims. One and only one dim can be -1
 * \return new symbol
 */
inline Symbol Reshape(Symbol data,
                      Shape shape = Shape()) {
  return Operator("Reshape")
      .SetParam("shape", shape)
      .SetInput("data", data)
      .CreateSymbol();
}

/*!
 * \breif Flatten input.
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to flatten. 
 * \return new symbol
 */
inline Symbol Flatten(const std::string& symbol_name,
                      Symbol data) {
  return Operator("Flatten")
      .SetInput("data", data)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs region-of-interest pooling on inputs.
 *        Resize bounding box coordinates by spatial_scale and crop input
 *        feature maps accordingly. The cropped feature maps are pooled by max
 *        pooling to a fixed size output indicated by pooled_size. batch_size
 *        will change to the number of region bounding boxes after ROIPooling
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to the pooling operator, a 4D Feature maps.
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]].
 *        (x1, y1) and (x2, y2) are top left and down right corners of
 *        designated region of interest. batch_index indicates the index of
 *        corresponding image in the input data
 * \param pooled_size fix pooled size: (h, w).
 * \param spatial_scale Ratio of input feature map height (or w) to raw image
 *        height (or w). Equals the reciprocal of total stride in convolutional layers
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
 * \breif Slice input equally along specified axis.
 * \param symbol_name name of the resulting symbol.
 * \param num_outputs Number of outputs to be sliced. 
 * \param axis Dimension along which to slice. 
 * \param squeeze_axis If true AND the sliced dimension becomes 1, squeeze that dimension. 
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
 * \breif Calculate Smooth L1 Loss(lhs, scalar).
 * \param symbol_name name of the resulting symbol.
 * \param lhs Left symbolic input to the function.
 * \param rhs Left symbolic input to the function.
 * \return new symbol
 */
inline Symbol smooth_l1(const std::string& symbol_name,
                        Symbol lhs,
                        Symbol rhs) {
  return Operator("smooth_l1")
      .SetInput("lhs", lhs)
      .SetInput("rhs", rhs)
      .CreateSymbol(symbol_name);
}

/*! \breif Softmax Mode.
 *        If set to instance, this operator will compute a softmax for each
 *        instance in the batch; this is the default mode. If set to channel,
 *        this operator will compute a num_channel-class softmax at each
 *        position of each instance; this can be used for fully convolutional
 *        network, image segmentation, etc.
 */
enum class SoftmaxActivationMode {
  channel = 0,
  instance = 1
};

/*!
 * \breif Apply softmax activation to input.
 *        This is intended for internal layers. For output (loss layer) please
 *        use SoftmaxOutput. If type=instance, this operator will compute a
 *        softmax for each instance in the batch; this is the default mode. If
 *        type=channel, this operator will compute a num_channel-class softmax
 *        at each position of each instance; this can be used for fully
 *        convolutional network, image segmentation, etc.
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to activation function. 
 * \param mode Softmax Mode.
 *        If set to instance, this operator will compute a softmax for each
 *        instance in the batch; this is the default mode. If set to channel,
 *        this operator will compute a num_channel-class softmax at each
 *        position of each instance; this can be used for fully convolutional
 *        network, image segmentation, etc.
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

/*! \breif If set to null, op will do nothing on output gradient.
 *        If set to batch, op will normalize gradient by divide batch sizeIf
 *        set to valid, op will normalize gradient by divide sample not ignored
 */
enum class SoftmaxOutputNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif Perform a softmax transformation on input, backprop with logloss. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to softmax. 
 * \param label Label data. 
 * \param grad_scale Scale the gradient by a float factor.
 * \param ignore_label the label value will be ignored during backward (only
 *        works if use_ignore is set to be true).
 * \param multi_output If set to true, for a (n,k,x_1,..,x_n) dimensional input
 *        tensor, softmax will generate n*x_1*...*x_n output, each has k classes
 * \param use_ignore If set to true, the ignore_label value will not contribute
 *        to the backward gradient
 * \param normalization If set to null, op will do nothing on output gradient.
 *        If set to batch, op will normalize gradient by divide batch sizeIf
 *        set to valid, op will normalize gradient by divide sample not ignored
 * \return new symbol
 */
inline Symbol SoftmaxOutput(const std::string& symbol_name,
                            Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::null) {
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
      .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
      .SetInput("data", data)
      .SetInput("label", label)
      .CreateSymbol(symbol_name);
}

/*! \breif If set to null, op will do nothing on output gradient.
 *        If set to batch, op will normalize gradient by divide batch sizeIf
 *        set to valid, op will normalize gradient by divide sample not ignored
 */
enum class SoftmaxNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif DEPRECATED: Perform a softmax transformation on input.
 *        Please use SoftmaxOutput
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to softmax. 
 * \param grad_scale Scale the gradient by a float factor.
 * \param ignore_label the label value will be ignored during backward (only
 *        works if use_ignore is set to be true).
 * \param multi_output If set to true, for a (n,k,x_1,..,x_n) dimensional input
 *        tensor, softmax will generate n*x_1*...*x_n output, each has k classes
 * \param use_ignore If set to true, the ignore_label value will not contribute
 *        to the backward gradient
 * \param normalization If set to null, op will do nothing on output gradient.
 *        If set to batch, op will normalize gradient by divide batch sizeIf
 *        set to valid, op will normalize gradient by divide sample not ignored
 * \return new symbol
 */
inline Symbol Softmax(const std::string& symbol_name,
                      Symbol data,
                      mx_float grad_scale = 1,
                      mx_float ignore_label = -1,
                      bool multi_output = false,
                      bool use_ignore = false,
                      SoftmaxNormalization normalization = SoftmaxNormalization::null) {
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
      .SetParam("normalization", SoftmaxNormalizationValues[int(normalization)])
      .SetInput("data", data)
      .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply swapaxis to input. 
 * \param symbol_name name of the resulting symbol.
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

/*! \breif upsampling method.
*/
enum class UpSamplingSampleType {
  bilinear = 0,
  nearest = 1
};

/*! \breif How to handle multiple input.
 *        concat means concatenate upsampled images along the channel
 *        dimension. sum means add all images together, only available for
 *        nearest neighbor upsampling.
 */
enum class UpSamplingMultiInputMode {
  concat = 0,
  sum = 1
};

/*!
 * \breif Perform nearest neighboor/bilinear up sampling to inputs.
 * \param symbol_name name of the resulting symbol.
 * \param data Array of tensors to upsample.
 * \param scale Up sampling scale.
 * \param sample_type upsampling method.
 * \param num_args Number of inputs to be upsampled.
 *        For nearest neighbor upsampling, this can be 1-N; the size of output
 *        will be(scale*h_0,scale*w_0) and all other inputs will be upsampled
 *        to thesame size. For bilinear upsampling this must be 2; 1 input and 1 weight.
 * \param num_filter Input filter.
 *        Only used by nearest sample_type.
 * \param multi_input_mode How to handle multiple input.
 *        concat means concatenate upsampled images along the channel
 *        dimension. sum means add all images together, only available for
 *        nearest neighbor upsampling.
 * \param workspace Tmp workspace for deconvolution (MB).
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

/*! \breif the type of RNN to compute.
*/
enum class RNNMode {
  gru = 0,
  lstm = 1,
  rnn_relu = 2,
  rnn_tanh = 3
};

/*!
 * \breif Apply a recurrent layer to input. 
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to RNN.
 * \param parameters Vector of all RNN trainable parameters.
 * \param state initial hidden state of the RNN.
 * \param state_cell initial cell state for LSTM networks (only for LSTM).
 * \param state_size size of the state for each layer.
 * \param num_layers number of stacked layers.
 * \param mode the type of RNN to compute.
 * \param bidirectional whether to use bidirectional recurrent layers.
 * \param p Fraction of the input that gets dropped out at training time.
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


/*!
 * \breif Apply a recurrent layer to input. 
 * \param data Input data to RNN.
 * \param parameters Vector of all RNN trainable parameters.
 * \param state initial hidden state of the RNN.
 * \param state_cell initial cell state for LSTM networks (only for LSTM).
 * \param state_size size of the state for each layer.
 * \param num_layers number of stacked layers.
 * \param mode the type of RNN to compute.
 * \param bidirectional whether to use bidirectional recurrent layers.
 * \param p Fraction of the input that gets dropped out at training time.
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

} //namespace cpp
} //namespace mxnet
#endif //ifndef _MXNETOP_H
