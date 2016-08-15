import mxnet as mx

def convolution_module(net, kernel_size, pad_size, filter_count, stride=(1, 1), work_space=2048, batch_norm=True, down_pool=False, up_pool=False, act_type="relu", convolution=True):
    if up_pool:
        net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count, workspace = work_space)
        net = mx.sym.BatchNorm(net)
        if act_type != "":
            net = mx.sym.Activation(net, act_type=act_type)

    if convolution:
        conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
        net = conv

    if batch_norm:
        net = mx.sym.BatchNorm(net)

    if act_type != "":
        net = mx.sym.Activation(net, act_type=act_type)

    if down_pool:
        pool = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
        net = pool

    return net


def get_symbol(num_classes = 10):
    data = mx.sym.Variable('data')
    kernel_size = (3, 3)
    pad_size = (1, 1)
    filter_count = 32
    pool1 = convolution_module(data, kernel_size, pad_size, filter_count=filter_count, down_pool=True)
    net = pool1
    pool2 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, down_pool=True)
    net = pool2
    pool3 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, down_pool=True)
    net = pool3
    pool4 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, down_pool=True)
    net = pool4
    net = mx.sym.Dropout(net)
    pool5 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 8, down_pool=True)
    net = pool5
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)

    # dirty "CROP" to wanted size... I was on old MxNet branch so used conv instead of crop for cropping
    net = convolution_module(net, (4, 4), (0, 0), filter_count=filter_count * 4)
    net = mx.sym.Concat(*[pool3, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)

    net = mx.sym.Concat(*[pool2, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
    convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = mx.sym.Concat(*[pool1, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, up_pool=True)
    net = mx.symbol.Flatten(net)
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes)
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    return net
