import mxnet as mx

def convolution_module(net, kernel_size, pad_size, filter_count, stride=(1, 1), work_space=2048, batch_norm=True, down_pool=False, up_pool=False, act_type="relu", convolution=True, name=''):
    if up_pool:
        net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count, workspace = work_space, name=('%s_deconv' % name))
        net = mx.sym.BatchNorm(net, name=('%s_bn' % name))
        if act_type != "":
            net = mx.sym.Activation(net, act_type=act_type, name=('%s_act' % name))

    if convolution:
        conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space, name=('%s_conv' % name))
        net = conv

    if batch_norm:
        net = mx.sym.BatchNorm(net, name=('%s_bn' % name))

    if act_type != "":
        net = mx.sym.Activation(net, act_type=act_type, name=('%s_act' % name))

    if down_pool:
        pool = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2), name=('%s_max_pool' % name))
        net = pool

    return net


def get_symbol():
    source = mx.sym.Variable("data")
    kernel_size = (3, 3)
    pad_size = (1, 1)
    filter_count = 32
    pool1 = convolution_module(source, kernel_size, pad_size, filter_count=filter_count, down_pool=True, name = "pool1")
    net = pool1
    pool2 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, down_pool=True, name = "pool2")
    net = pool2
    pool3 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, down_pool=True, name = "pool3")
    net = pool3
    pool4 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, down_pool=True, name = "pool4")
    net = pool4
    net = mx.sym.Dropout(net, name = "pool4_drop")
    pool5 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 8, down_pool=True, name = "pool5")
    net = pool5
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True, name = "pool5_conv1")
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True, name = "pool5_conv2")

    # dirty "CROP" to wanted size... I was on old MxNet branch so user conv instead of crop for cropping
    net = convolution_module(net, (4, 4), (0, 0), filter_count=filter_count * 4, name = "pool5_conv3")
    net = mx.sym.Concat(*[pool3, net], name = "pool3_concat")
    net = mx.sym.Dropout(net, name = "pool3_drop")
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, name = "pool3_conv1")
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True, name = "pool3_conv2")


    net = mx.sym.Concat(*[pool2, net], name = "pool2_concat")
    net = mx.sym.Dropout(net, name = "pool2_drop")
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, name = "pool2_conv1")
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True, name = "pool2_conv2")
    net = mx.sym.Concat(*[pool1, net], name = "pool1_concat")
    net = mx.sym.Dropout(net, name = "pool1_drop")
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, name = "pool1_conv1")
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, up_pool=True, name = "pool1_conv2")

    net = convolution_module(net, kernel_size, pad_size, filter_count=1, batch_norm=False, act_type="", name = "pool1_conv3")

    net = mx.symbol.Flatten(net, name = "flatten")
    return mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
