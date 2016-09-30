library(mxnet)

Conv <- function(data, num_filter, kernel = c(1, 1), stride = c(1, 1),
                 pad = c(0, 0), name = '', suffix = '') {
    conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel,
                                 stride = stride, pad = pad, no_bias = TRUE,
                                 name = paste('conv_', name, suffix, sep = ''))
    
    bn = mx.symbol.BatchNorm(data = conv, name = paste('bn_', name, suffix, sep = '') ,
                             fix_gamma = TRUE)
    
    act = mx.symbol.Activation(data = bn, act_type = 'relu',
                               name = paste('relu_', name, suffix, sep = ''))
    
    return(act)
}

Inception7A <- function(data, num_1x1, num_3x3_red, num_3x3_1, num_3x3_2,
                        num_5x5_red, num_5x5, pool, proj, name) {
  tower_1x1 = Conv(data, num_1x1, name = paste(name, '_conv', sep = ''))
  
  tower_5x5 = Conv(data, num_5x5_red, name = paste(name, '_tower', sep = ''), suffix = '_conv')
  
  tower_5x5 = Conv(tower_5x5, num_5x5, kernel = c(5, 5), pad = c(2, 2),
                   name = paste(name, '_tower', sep = ''), suffix = '_conv_1')
  
  tower_3x3 = Conv(data, num_3x3_red, name = paste(name, '_tower_1', sep = ''), suffix = '_conv')
  
  tower_3x3 = Conv(tower_3x3, num_3x3_1, kernel = c(3, 3), pad = c(1, 1),
                   name = paste(name, '_tower_1', sep = ''), suffix = '_conv_1')
  
  tower_3x3 = Conv(tower_3x3, num_3x3_2, kernel = c(3, 3), pad = c(1, 1),
                   name = paste(name, '_tower_1', sep = ''), suffix = '_conv_2')
  
  pooling = mx.symbol.Pooling(data = data, kernel = c(3, 3), stride = c(1, 1),
                              pad = c(1, 1), pool_type = pool,
                              name = paste(pool, '_pool_', name, '_pool', sep = ''))
  
  cproj = Conv(pooling, proj, name = paste(name, '_tower_2', sep = ''), suffix = '_conv')
  
  concat = mx.symbol.Concat(c(tower_1x1, tower_5x5, tower_3x3, cproj),
                            num.args =  4, name = paste('ch_concat_', name, '_chconcat', sep = ''))
  return(concat)
}

Inception7B <- function(data, num_3x3, num_d3x3_red, num_d3x3_1, num_d3x3_2, pool, name) {
  tower_3x3 = Conv(data, num_3x3, kernel = c(3, 3), pad = c(0, 0),
                   stride = c(2, 2), name = paste(name, '_conv', sep = ''))
  
  tower_d3x3 = Conv(data, num_d3x3_red, name = paste(name, '_tower', sep = ''), suffix = '_conv')
  
  tower_d3x3 = Conv(tower_d3x3, num_d3x3_1, kernel = c(3, 3), pad = c(1, 1),
                    stride = c(1, 1), name = paste(name, '_tower', sep = ''), suffix = '_conv_1')
  
  tower_d3x3 = Conv(tower_d3x3, num_d3x3_2, kernel = c(3, 3), pad = c(0, 0),
                    stride = c(2, 2), name = paste(name, '_tower', sep = ''), suffix = '_conv_2')
  
  pooling = mx.symbol.Pooling(data = data, kernel = c(3, 3), stride = c(2, 2),
                              pad = c(0, 0), pool_type = "max",
                              name = paste('max_pool_', name, '_pool', sep = ''))
  
  concat = mx.symbol.Concat(c(tower_3x3, tower_d3x3, pooling),
                            num.args = 3,
                            name = paste('ch_concat_', name, '_chconcat', sep = ''))
  return(concat)
}

Inception7C <- function(data, num_1x1, num_d7_red, num_d7_1, num_d7_2,
                        num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4,
                        pool, proj, name) {
  tower_1x1 = Conv(data = data, num_filter = num_1x1,
                   kernel = c(1, 1), name = paste(name, '_conv', sep = ''))
  
  tower_d7 = Conv(data = data, num_filter = num_d7_red,
                  name = paste(name, '_tower', sep = ''), suffix = '_conv')
  
  tower_d7 = Conv(data = tower_d7, num_filter = num_d7_1,
                  kernel = c(7, 1), pad = c(3, 0), 
                  name = paste(name, '_tower', sep = ''), suffix = '_conv_1')
  
  tower_d7 = Conv(data = tower_d7, num_filter = num_d7_2,
                  kernel = c(1, 7), pad = c(0, 3), 
                  name = paste(name, '_tower', sep = ''), suffix = '_conv_2')
  
  tower_q7 = Conv(data = data, num_filter = num_q7_red,
                  name = paste(name, '_tower_1', sep = ''), suffix = '_conv')
  
  tower_q7 = Conv(data = tower_q7, num_filter = num_q7_1,
                  kernel = c(1, 7), pad = c(0, 3), 
                  name = paste(name, '_tower_1', sep = ''), suffix = '_conv_1')
  
  tower_q7 = Conv(data = tower_q7, num_filter = num_q7_2,
                  kernel = c(7, 1), pad = c(3, 0),
                  name = paste(name, '_tower_1', sep = ''), suffix = '_conv_2')
  
  tower_q7 = Conv(data = tower_q7, num_filter = num_q7_3,
                  kernel = c(1, 7), pad = c(0, 3),
                  name = paste(name, '_tower_1', sep = ''), suffix = '_conv_3')
  
  tower_q7 = Conv(data = tower_q7, num_filter = num_q7_4,
                  kernel = c(7, 1), pad = c(3, 0),
                  name = paste(name, '_tower_1', sep = ''), suffix = '_conv_4')
  
  pooling = mx.symbol.Pooling(data = data, kernel = c(3, 3),
                              stride = c(1, 1), pad = c(1, 1), pool_type = pool,
                              name = paste(pool, '_pool_', name, '_pool', sep = ''))
  
  cproj = Conv(data = pooling, num_filter = proj, kernel = c(1, 1),
               name = paste(name, '_tower_2', sep = ''), suffix = '_conv')
  
  concat = mx.symbol.Concat(c(tower_1x1, tower_d7, tower_q7, cproj),
                            num.args = 4,
                            name = paste('ch_concat_', name, '_chconcat', sep = ''))
  return(concat)
}

Inception7D <- function(data, num_3x3_red, num_3x3, num_d7_3x3_red,
                        num_d7_1, num_d7_2, num_d7_3x3, pool, name) {
  tower_3x3 = Conv(data = data, num_filter = num_3x3_red,
                   name = paste(name, '_tower', sep = ''), suffix = '_conv')
  
  tower_3x3 = Conv(data = tower_3x3, num_filter = num_3x3,
                   kernel = c(3, 3), pad = c(0, 0),
                   stride = c(2, 2), name = paste(name, '_tower', sep = ''),
                   suffix = '_conv_1')
  
  tower_d7_3x3 = Conv(data = data, num_filter = num_d7_3x3_red,
                      name = paste(name, '_tower_1', sep = ''), suffix = '_conv')
  
  tower_d7_3x3 = Conv(data = tower_d7_3x3, num_filter = num_d7_1,
                      kernel = c(7, 1), pad = c(3, 0), 
                      name = paste(name, '_tower_1', sep = ''), suffix = '_conv_1')
  
  tower_d7_3x3 = Conv(data = tower_d7_3x3, num_filter = num_d7_2,
                      kernel = c(1, 7), pad = c(0, 3),
                      name = paste(name, '_tower_1', sep = ''), suffix = '_conv_2')
  
  tower_d7_3x3 = Conv(data = tower_d7_3x3, num_filter = num_d7_3x3,
                      kernel = c(3, 3), stride = c(2, 2),
                      name = paste(name, '_tower_1', sep = ''), suffix = '_conv_3')
  
  pooling = mx.symbol.Pooling(data = data, kernel = c(3, 3), stride = c(2, 2),
                              pool_type = pool, name = paste(pool, '_pool_', name, '_pool', sep = ''))
  
  concat = mx.symbol.Concat(c(tower_3x3, tower_d7_3x3, pooling),
                            num.args = 3, 
                            name = paste('ch_concat_', name, '_chconcat', sep = ''))
  return(concat)
}

Inception7E <- function(data, num_1x1, num_d3_red, num_d3_1, num_d3_2,
                        num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2,
                        pool, proj, name) {
  tower_1x1 = Conv(data = data, num_filter = num_1x1,
                   kernel = c(1, 1), name = paste(name, '_conv', sep = ''))
  
  tower_d3 = Conv(data = data, num_filter = num_d3_red,
                  name = paste(name, '_tower', sep = ''), suffix = '_conv')
  
  tower_d3_a = Conv(data = tower_d3, num_filter = num_d3_1,
                    kernel = c(3, 1), pad = c(1, 0),
                    name = paste(name, '_tower', sep = ''),
                    suffix = '_mixed_conv')
  
  tower_d3_b = Conv(data = tower_d3, num_filter = num_d3_2,
                    kernel = c(1, 3), pad = c(0, 1),
                    name = paste(name, '_tower', sep = ''), 
                    suffix = '_mixed_conv_1')
  
  tower_3x3_d3 = Conv(data = data, num_filter = num_3x3_d3_red,
                      name = paste(name, '_tower_1', sep = ''),
                      suffix = '_conv')
  
  tower_3x3_d3 = Conv(data = tower_3x3_d3, num_filter = num_3x3,
                      kernel = c(3, 3), pad = c(1, 1),
                      name = paste(name, '_tower_1', sep = ''), 
                      suffix = '_conv_1')
  
  tower_3x3_d3_a = Conv(data = tower_3x3_d3, num_filter = num_3x3_d3_1,
                        kernel = c(3, 1), pad = c(1, 0),
                        name = paste(name, '_tower_1', sep = ''), 
                        suffix = '_mixed_conv')
  
  tower_3x3_d3_b = Conv(data = tower_3x3_d3, num_filter = num_3x3_d3_2,
                        kernel = c(1, 3), pad = c(0, 1),
                        name = paste(name, '_tower_1', sep = ''),
                        suffix = '_mixed_conv_1')
  
  pooling = mx.symbol.Pooling(data = data, kernel = c(3, 3),
                              stride = c(1, 1), pad = c(1, 1),
                              pool_type = pool,     
                              name = paste(pool, '_pool_', name, '_pool', sep = ''))
  
  cproj = Conv(data = pooling, num_filter = proj,
               kernel = c(1, 1), name = paste(name, '_tower_2', sep = ''),
               suffix = '_conv')
  
  concat = mx.symbol.Concat(c(tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj),
                            num.args = 6, name = paste('ch_concat_', name, '_chconcat', sep = ''))
  return(concat)
}

get_symbol <- function(num_classes = 1000) {
  data = mx.symbol.Variable(name = "data")
  # stage 1
  conv = Conv(data, 32, kernel = c(3, 3), stride = c(2, 2), name = "conv")
  conv_1 = Conv(conv, 32, kernel = c(3, 3), name = "conv_1")
  conv_2 = Conv(conv_1, 64, kernel = c(3, 3), pad = c(1, 1), name = "conv_2")
  pool = mx.symbol.Pooling(data = conv_2, kernel = c(3, 3), stride = c(2, 2),
                           pool_type = "max", name = "pool")
  # stage 2
  conv_3 = Conv(pool, 80, kernel = c(1, 1), name = "conv_3")
  conv_4 = Conv(conv_3, 192, kernel = c(3, 3), name = "conv_4")
  pool1 = mx.symbol.Pooling(data = conv_4, kernel = c(3, 3),
                            stride = c(2, 2), pool_type = "max", name = "pool1")
  # stage 3
  in3a = Inception7A(pool1, 64, 64, 96, 96, 48, 64, "avg", 32, "mixed")
  in3b = Inception7A(in3a, 64, 64, 96, 96, 48, 64, "avg", 64, "mixed_1")
  in3c = Inception7A(in3b, 64, 64, 96, 96, 48, 64, "avg", 64, "mixed_2")
  in3d = Inception7B(in3c, 384, 64, 96, 96, "max", "mixed_3")
  # stage 4
  in4a = Inception7C(in3d, 192, 128, 128, 192, 128, 128, 128, 128, 192, "avg", 192, "mixed_4")
  in4b = Inception7C(in4a, 192, 160, 160, 192, 160, 160, 160, 160, 192, "avg", 192, "mixed_5")
  in4c = Inception7C(in4b, 192, 160, 160, 192, 160, 160, 160, 160, 192, "avg", 192, "mixed_6")
  in4d = Inception7C(in4c, 192, 192, 192, 192, 192, 192, 192, 192, 192, "avg", 192, "mixed_7")
  in4e = Inception7D(in4d, 192, 320, 192, 192, 192, 192, "max", "mixed_8")
  # stage 5
  in5a = Inception7E(in4e, 320, 384, 384, 384, 448, 384, 384, 384, "avg", 192, "mixed_9")
  in5b = Inception7E(in5a, 320, 384, 384, 384, 448, 384, 384, 384, "max", 192, "mixed_10")
  # pool
  pool = mx.symbol.Pooling(data = in5b, kernel = c(8, 8),
                           stride = c(1, 1), pool_type = "avg", name = "global_pool")
  flatten = mx.symbol.Flatten(data = pool, name = "flatten")
  fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = num_classes, name = 'fc1')
  softmax = mx.symbol.SoftmaxOutput(data = fc1, name = 'softmax')
  return(softmax)
}
