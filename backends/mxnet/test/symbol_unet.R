library(mxnet)

convolution_module <- function(net, kernel_size, pad_size,
                               filter_count, stride = c(1, 1), work_space = 2048,
                               batch_norm = TRUE, down_pool = FALSE, up_pool = FALSE,
                               act_type = "relu", convolution = TRUE, name = '') {
    if (up_pool) {
      net = mx.symbol.Deconvolution(net, kernel = c(2, 2), pad = c(0, 0),
                                    stride = c(2, 2), num_filter = filter_count,
                                    workspace = work_space,
                                    name = paste(name, "_deconv", sep = ''))
      net = mx.symbol.BatchNorm(net, name = paste(name, "_bn", sep = ''))
      if (act_type != "") {
        net = mx.symbol.Activation(net, act_type = act_type, name = paste(name, "_act", sep = ''))
      }
    }
    if (convolution) {
      conv = mx.symbol.Convolution(data = net, kernel = kernel_size, stride = stride,
                                   pad = pad_size, num_filter = filter_count, workspace = work_space,
                                   name = paste(name, "_conv", sep =''))
      net = conv
    }
    
    if (batch_norm) {
      net = mx.symbol.BatchNorm(net, name = paste(name, "_bn", sep = ''))
    }
    
    if (act_type != "") {
      net = mx.symbol.Activation(net, act_type = act_type, name = paste(name, "_act", sep = ''))
    }
    
    if (down_pool) {
      pool = mx.symbol.Pooling(net, pool_type = "max", kernel = c(2, 2), stride = c(2, 2),
                               name = paste(name, "_max_pool", sep = ''))
      net = pool
    }
    return(net)
}

get_symbol <- function() {
  data = mx.symbol.Variable('data')
  kernel_size = c(3, 3)
  pad_size = c(1, 1)
  filter_count = 32
  pool1 = convolution_module(data, kernel_size, pad_size, filter_count = filter_count, down_pool = TRUE, name = "pool1")
  net = pool1
  pool2 = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 2, down_pool = TRUE, name = "pool2")
  net = pool2
  pool3 = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, down_pool = TRUE, name = "pool3")
  net = pool3
  pool4 = convolution_module(net,
                             kernel_size,
                             pad_size,
                             filter_count = filter_count * 4,
                             down_pool = TRUE, name = "pool4")
  net = pool4
  net = mx.symbol.Dropout(net, name = "pool4_drop")
  pool5 = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 8, down_pool = TRUE, name = "pool5")
  net = pool5
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, up_pool = TRUE, name = "pool5_conv1")
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, up_pool = TRUE, name = "pool5_conv2")
  
  # dirty "CROP" to wanted size... I was on old MxNet branch so used conv instead of crop for cropping
  net = convolution_module(net, c(4, 4), c(0, 0), filter_count = filter_count * 4, name = "pool5_conv3")
  
  net = mx.symbol.Concat(c(pool3, net), num.args = 2, name = "pool3_concat")
  net = mx.symbol.Dropout(net, name = "pool3_drop")
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, name = "pool3_conv1")
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4,
                           up_pool = TRUE, name = "pool3_conv2")
  
  net = mx.symbol.Concat(c(pool2, net), num.args = 2, name = "pool2_concat")
  net = mx.symbol.Dropout(net, name = "pool2_drop")
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, name = "pool2_conv1")
  net = convolution_module(net, kernel_size, pad_size,
                           filter_count = filter_count * 4, up_pool = TRUE, name = "pool2_conv2")
  convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4)
  net = mx.symbol.Concat(c(pool1, net), num.args = 2, name = "pool1_concat")
  net = mx.symbol.Dropout(net, name = "pool1_drop")
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 2, name = "pool1_conv1")
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 2,
                           up_pool = TRUE, name = "pool1_conv2")
  net = convolution_module(net, kernel_size, pad_size, filter_count = 1,
                           batch_norm = FALSE, act_type = "", name = "pool1_conv3")
  net = mx.symbol.Flatten(net, name = "flatten")
  net = mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
  
  return(net)
}
