#OperatorPropertyReg
* Name: `Activation`
* Description: Apply activation function to input.Softmax Activation is only available with CUDNN on GPUand will be computed at each location across channel if input is 4D.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to activation function.|
|act_type|{'relu', 'sigmoid', 'softrelu', 'tanh'}, required|Activation function to be applied.|

* Name: `BatchNorm`
* Description: Apply batch normalization to input.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to batch normalization|
|eps|float, optional, default=0.001|Epsilon to prevent div 0|
|momentum|float, optional, default=0.9|Momentum for moving average|
|fix_gamma|boolean, optional, default=True|Fix gamma while training|
|use_global_stats|boolean, optional, default=False|Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.|

* Name: `BlockGrad`
* Description: Get output from a symbol and pass 0 gradient back
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data.|

* Name: `sum`
* Description: Take sum of the src in the given axis and returns a NDArray. Follows numpy semantics.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|
|axis|Shape(tuple), optional, default=()|Same as Numpy. The axes to perform the reduction.If left empty, a global reduction will be performed.|
|keepdims|boolean, optional, default=False|Same as Numpy. If keepdims is set to true, the axis which is reduced is left in the result as dimension with size one.|

* Name: `sum_axis`
* Description: (Depreciated! Use sum instead!) Take sum of the src in the given axis and returns a NDArray. Follows numpy semantics.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|
|axis|Shape(tuple), optional, default=()|Same as Numpy. The axes to perform the reduction.If left empty, a global reduction will be performed.|
|keepdims|boolean, optional, default=False|Same as Numpy. If keepdims is set to true, the axis which is reduced is left in the result as dimension with size one.|

* Name: `broadcast_axis`
* Description: Broadcast data in the given axis to the given size. The original size of the broadcasting axis must be 1.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|
|axis|Shape(tuple), optional, default=()|The axes to perform the broadcasting.|
|size|Shape(tuple), optional, default=()|Target sizes of the broadcasting axes.|

* Name: `broadcast_to`
* Description: Broadcast data to the target shape. The original size of the broadcasting axis must be 1.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|
|shape|Shape(tuple), optional, default=()|The shape of the desired array. We can set the dim to zero if it's same as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.|

* Name: `Cast`
* Description: Cast array to a different data type.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to cast function.|
|dtype|{'float16', 'float32', 'float64', 'int32', 'uint8'}, required|Target data type.|

* Name: `Concat`
* Description: Perform an feature concat on channel dim (defaut is 1) over all
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol[]|List of tensors to concatenate|
|num_args|int, required|Number of inputs to be concated.|
|dim|int, optional, default='1'|the dimension to be concated.|

* Name: `Convolution`
* Description: Apply convolution to input then add a bias.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the ConvolutionOp.|
|weight|Symbol|Weight matrix.|
|bias|Symbol|Bias parameter.|
|kernel|Shape(tuple), required|convolution kernel size: (y, x)|
|stride|Shape(tuple), optional, default=(1,1)|convolution stride: (y, x)|
|dilate|Shape(tuple), optional, default=(1,1)|convolution dilate: (y, x)|
|pad|Shape(tuple), optional, default=(0,0)|pad for convolution: (y, x)|
|num_filter|int (non-negative), required|convolution filter(channel) number|
|num_group|int (non-negative), optional, default=1|Number of groups partition. This option is not supported by CuDNN, you can use SliceChannel to num_group,apply convolution and concat instead to achieve the same need.|
|workspace|long (non-negative), optional, default=512|Tmp workspace for convolution (MB).|
|no_bias|boolean, optional, default=False|Whether to disable bias parameter.|
|cudnn_tune|{'fastest', 'limited_workspace', 'off'},optional, default='limited_workspace'|Whether to find convolution algo by running performance test.Leads to higher startup time but may give better speed|

* Name: `Crop`
* Description: Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or with width and height of the second input symbol, i.e., with one input, we need h_w to specify the crop height and width, otherwise the second input symbol's size will be used
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol or Symbol[]|Tensor or List of Tensors, the second input will be used as crop_like shape reference|
|num_args|int, required|Number of inputs for crop, if equals one, then we will use the h_wfor crop height and width, else if equals two, then we will use the heightand width of the second input symbol, we name crop_like here|
|offset|Shape(tuple), optional, default=(0,0)|crop offset coordinate: (y, x)|
|h_w|Shape(tuple), optional, default=(0,0)|crop height and weight: (h, w)|
|center_crop|boolean, optional, default=False|If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like|

* Name: `_CrossDeviceCopy`
* Description: Special op to copy data cross device

* Name: `Custom`
* Description: Custom operator implemented in frontend.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|op_type|string|Type of custom operator. Must be registered first.|

* Name: `Deconvolution`
* Description: Apply deconvolution to input then add a bias.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the DeconvolutionOp.|
|weight|Symbol|Weight matrix.|
|bias|Symbol|Bias parameter.|
|kernel|Shape(tuple), required|deconvolution kernel size: (y, x)|
|stride|Shape(tuple), optional, default=(1,1)|deconvolution stride: (y, x)|
|pad|Shape(tuple), optional, default=(0,0)|pad for deconvolution: (y, x), a good number is : (kernel-1)/2, if target_shape set, pad will be ignored and will be computed automatically|
|adj|Shape(tuple), optional, default=(0,0)|adjustment for output shape: (y, x), if target_shape set, adj will be ignored and will be computed automatically|
|target_shape|Shape(tuple), optional, default=(0,0)|output shape with targe shape : (y, x)|
|num_filter|int (non-negative), required|deconvolution filter(channel) number|
|num_group|int (non-negative), optional, default=1|number of groups partition|
|workspace|long (non-negative), optional, default=512|Tmp workspace for deconvolution (MB)|
|no_bias|boolean, optional, default=True|Whether to disable bias parameter.|

* Name: `Dropout`
* Description: Apply dropout to input
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to dropout.|
|p|float, optional, default=0.5|Fraction of the input that gets dropped out at training time|

* Name: `broadcast_plus`
* Description: lhs add rhs with broadcast
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `broadcast_minus`
* Description: lhs minus rhs with broadcast
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `broadcast_mul`
* Description: lhs multiple rhs with broadcast
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `broadcast_div`
* Description: lhs divide rhs with broadcast
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `broadcast_power`
* Description: lhs power rhs with broadcast
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `_Plus`
* Description: Add lhs and rhs
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `_Minus`
* Description: Minus lhs and rhs
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `_Mul`
* Description: Multiply lhs and rhs
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `_Div`
* Description: Multiply lhs by rhs
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `_Power`
* Description: Elementwise power(lhs, rhs)
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `_Maximum`
* Description: Elementwise max of lhs by rhs
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `_Minimum`
* Description: Elementwise min of lhs by rhs
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `_PlusScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `_MinusScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `_RMinusScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `_MulScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `_DivScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `_RDivScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `_MaximumScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `_MinimumScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `_PowerScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `_RPowerScalar`
* Description: 
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `ElementWiseSum`
* Description: Perform an elementwise sum over all the inputs.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|num_args|int, required|Number of inputs to be summed.|

* Name: `abs`
* Description: Take absolute value of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `sign`
* Description: Take sign value of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `round`
* Description: Take round value of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `ceil`
* Description: Take ceil value of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `floor`
* Description: Take floor value of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `square`
* Description: Take square of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `sqrt`
* Description: Take sqrt of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `rsqrt`
* Description: Take rsqrt of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `exp`
* Description: Take exp of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `log`
* Description: Take log of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `cos`
* Description: Take cos of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `sin`
* Description: Take sin of the src
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `Embedding`
* Description: Get embedding for one-hot input. A n-dimensional input tensor will be trainsformed into a (n+1)-dimensional tensor, where a new dimension is added for the embedding results.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the EmbeddingOp.|
|weight|Symbol|Enbedding weight matrix.|
|input_dim|int, required|input dim of one-hot encoding|
|output_dim|int, required|output dim of embedding|

* Name: `FullyConnected`
* Description: Apply matrix multiplication to input then add a bias.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the FullyConnectedOp.|
|weight|Symbol|Weight matrix.|
|bias|Symbol|Bias parameter.|
|num_hidden|int, required|Number of hidden nodes of the output.|
|no_bias|boolean, optional, default=False|Whether to disable bias parameter.|

* Name: `IdentityAttachKLSparseReg`
* Description: Apply a sparse regularization to the output a sigmoid activation function.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data.|
|sparseness_target|float, optional, default=0.1|The sparseness target|
|penalty|float, optional, default=0.001|The tradeoff parameter for the sparseness penalty|
|momentum|float, optional, default=0.9|The momentum for running average|

* Name: `L2Normalization`
* Description: Set the l2 norm of each instance to a constant.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the L2NormalizationOp.|
|eps|float, optional, default=1e-10|Epsilon to prevent div 0|

* Name: `LeakyReLU`
* Description: Apply activation function to input.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to activation function.|
|act_type|{'elu', 'leaky', 'prelu', 'rrelu'},optional, default='leaky'|Activation function to be applied.|
|slope|float, optional, default=0.25|Init slope for the activation. (For leaky and elu only)|
|lower_bound|float, optional, default=0.125|Lower bound of random slope. (For rrelu only)|
|upper_bound|float, optional, default=0.334|Upper bound of random slope. (For rrelu only)|

* Name: `softmax_cross_entropy`
* Description: Calculate cross_entropy(lhs, one_hot(rhs))
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `LRN`
* Description: Apply convolution to input then add a bias.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the ConvolutionOp.|
|alpha|float, optional, default=0.0001|value of the alpha variance scaling parameter in the normalization formula|
|beta|float, optional, default=0.75|value of the beta power parameter in the normalization formula|
|knorm|float, optional, default=2|value of the k parameter in normalization formula|
|nsize|int (non-negative), required|normalization window width in elements.|

* Name: `MakeLoss`
* Description: Get output from a symbol and pass 1 gradient back. This is used as a terminal loss if unary and binary operator are used to composite a loss with no declaration of backward dependency
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data.|
|grad_scale|float, optional, default=1|gradient scale as a supplement to unary and binary operators|

* Name: `transpose`
* Description: Transpose the input matrix and return a new one
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|
|axes|Shape(tuple), optional, default=()|Target axis order. By default the axes will be inverted.|

* Name: `expand_dims`
* Description: Expand the shape of array by inserting a new axis.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|
|axis|int (non-negative), required|Position (amongst axes) where new axis is to be inserted.|

* Name: `slice_axis`
* Description: Slice the input along certain axis and return a sliced array.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|
|axis|int, required|The axis to be sliced|
|begin|int, required|The beginning index to be sliced|
|end|int, required|The end index to be sliced|

* Name: `dot`
* Description: Calculate dot product of two matrices or two vectors
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `batch_dot`
* Description: Calculate batched dot product of two matrices. (batch, M, K) batch_dot (batch, K, N) --> (batch, M, N)
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|lhs|Symbol|Left symbolic input to the function|
|rhs|Symbol|Right symbolic input to the function|

* Name: `_Native`
* Description: Stub for implementing an operator implemented in native frontend language.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|info|, required||
|need_top_grad|boolean, optional, default=True|Whether this layer needs out grad for backward. Should be false for loss layers.|

* Name: `_NDArray`
* Description: Stub for implementing an operator implemented in native frontend language with ndarray.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|info|, required||

* Name: `Pooling`
* Description: Perform spatial pooling on inputs.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the pooling operator.|
|global_pool|boolean, optional, default=False|Ignore kernel size, do global pooling based on current input feature map. This is useful for input with different shape|
|kernel|Shape(tuple), required|pooling kernel size: (y, x)|
|pool_type|{'avg', 'max', 'sum'}, required|Pooling type to be applied.|
|stride|Shape(tuple), optional, default=(1,1)|stride: for pooling (y, x)|
|pad|Shape(tuple), optional, default=(0,0)|pad for pooling: (y, x)|

* Name: `LinearRegressionOutput`
* Description: Use linear regression for final output, this is used on final output of a net.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to function.|
|label|Symbol|Input label to function.|
|grad_scale|float, optional, default=1|Scale the gradient by a float factor|

* Name: `MAERegressionOutput`
* Description: Use mean absolute error regression for final output, this is used on final output of a net.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to function.|
|label|Symbol|Input label to function.|
|grad_scale|float, optional, default=1|Scale the gradient by a float factor|

* Name: `LogisticRegressionOutput`
* Description: Use Logistic regression for final output, this is used on final output of a net.
Logistic regression is suitable for binary classification or probability prediction tasks.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to function.|
|label|Symbol|Input label to function.|
|grad_scale|float, optional, default=1|Scale the gradient by a float factor|

* Name: `Reshape`
* Description: Reshape input to target shape
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to reshape.|
|target_shape|Shape(tuple), optional, default=(0,0)|(Deprecated! Use shape instead.) Target new shape. One and only one dim can be 0, in which case it will be inferred from the rest of dims|
|keep_highest|boolean, optional, default=False|(Deprecated! Use shape instead.) Whether keep the highest dim unchanged.If set to yes, than the first dim in target_shape is ignored,and always fixed as input|
|shape|, optional, default=()|Target new shape. If the dim is same, set it to 0. If the dim is set to be -1, it will be inferred from the rest of dims. One and only one dim can be -1|

* Name: `Flatten`
* Description: Flatten input
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to flatten.|

* Name: `ROIPooling`
* Description: Performs region-of-interest pooling on inputs. Resize bounding box coordinates by spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled by max pooling to a fixed size output indicated by pooled_size. batch_size will change to the number of region bounding boxes after ROIPooling
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the pooling operator, a 4D Feature maps|
|rois|Symbol|Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners of designated region of interest. batch_index indicates the index of corresponding image in the input data|
|pooled_size|Shape(tuple), required|fix pooled size: (h, w)|
|spatial_scale|float, required|Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers|

* Name: `uniform`
* Description: Sample a uniform distribution
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|low|float, optional, default=0|The lower bound of distribution|
|high|float, optional, default=1|The upper bound of distribution|
|shape|Shape(tuple), required|The shape of the output|

* Name: `normal`
* Description: Sample a normal distribution
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|loc|float, optional, default=0|Mean of the distribution.|
|scale|float, optional, default=1|Standard deviation of the distribution.|
|shape|Shape(tuple), required|The shape of the output|

* Name: `SliceChannel`
* Description: Slice input equally along specified axis
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|num_outputs|int, required|Number of outputs to be sliced.|
|axis|int, optional, default='1'|Dimension along which to slice.|
|squeeze_axis|boolean, optional, default=False|If true AND the sliced dimension becomes 1, squeeze that dimension.|

* Name: `smooth_l1`
* Description: Calculate Smooth L1 Loss(lhs, scalar)
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|src|Symbol|Left symbolic input to the function|

* Name: `SoftmaxActivation`
* Description: Apply softmax activation to input. This is intended for internal layers. For output (loss layer) please use SoftmaxOutput. If type=instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If type=channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to activation function.|
|mode|{'channel', 'instance'},optional, default='instance'|Softmax Mode. If set to instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If set to channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.|

* Name: `SoftmaxOutput`
* Description: Perform a softmax transformation on input, backprop with logloss.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to softmax.|
|label|Symbol|Label data, can also be probability value with same shape as data|
|grad_scale|float, optional, default=1|Scale the gradient by a float factor|
|ignore_label|float, optional, default=-1|the label value will be ignored during backward (only works if use_ignore is set to be true).|
|multi_output|boolean, optional, default=False|If set to true, for a (n,k,x_1,..,x_n) dimensional input tensor, softmax will generate n*x_1*...*x_n output, each has k classes|
|use_ignore|boolean, optional, default=False|If set to true, the ignore_label value will not contribute to the backward gradient|
|normalization|{'batch', 'null', 'valid'},optional, default='null'|If set to null, op will do nothing on output gradient.If set to batch, op will normalize gradient by divide batch sizeIf set to valid, op will normalize gradient by divide sample not ignored|

* Name: `Softmax`
* Description: DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to softmax.|
|grad_scale|float, optional, default=1|Scale the gradient by a float factor|
|ignore_label|float, optional, default=-1|the label value will be ignored during backward (only works if use_ignore is set to be true).|
|multi_output|boolean, optional, default=False|If set to true, for a (n,k,x_1,..,x_n) dimensional input tensor, softmax will generate n*x_1*...*x_n output, each has k classes|
|use_ignore|boolean, optional, default=False|If set to true, the ignore_label value will not contribute to the backward gradient|
|normalization|{'batch', 'null', 'valid'},optional, default='null'|If set to null, op will do nothing on output gradient.If set to batch, op will normalize gradient by divide batch sizeIf set to valid, op will normalize gradient by divide sample not ignored|

* Name: `SpatialTransformer`
* Description: Apply spatial transformer to input feature map.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the SpatialTransformerOp.|
|loc|Symbol|localisation net, the output dim should be 6 when transform_type is affine, and the name of loc symbol should better starts with 'stn_loc', so that initialization it with iddentify tranform, or you shold initialize the weight and bias by yourself.|
|target_shape|Shape(tuple), optional, default=(0,0)|output shape(h, w) of spatial transformer: (y, x)|
|transform_type|{'affine'}, required|transformation type|
|sampler_type|{'bilinear'}, required|sampling type|

* Name: `SwapAxis`
* Description: Apply swapaxis to input.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol|Input data to the SwapAxisOp.|
|dim1|int (non-negative), optional, default=0|the first axis to be swapped.|
|dim2|int (non-negative), optional, default=0|the second axis to be swapped.|

* Name: `UpSampling`
* Description: Perform nearest neighboor/bilinear up sampling to inputs
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data|Symbol[]|Array of tensors to upsample|
|scale|int (non-negative), required|Up sampling scale|
|num_filter|int (non-negative), optional, default=0|Input filter. Only used by nearest sample_type.|
|sample_type|{'bilinear', 'nearest'}, required|upsampling method|
|multi_input_mode|{'concat', 'sum'},optional, default='concat'|How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.|
|num_args|int, required|Number of inputs to be upsampled. For nearest neighbor upsampling, this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other inputs will be upsampled to thesame size. For bilinear upsampling this must be 2; 1 input and 1 weight.|
|workspace|long (non-negative), optional, default=512|Tmp workspace for deconvolution (MB)|

#OptimizerReg
* Name: `ccsgd`
* Description: Stochastic gradient decent optimizer implemented in C++.

#DataIteratorReg
* Name: `CSVIter`
* Description: Create iterator for dataset in csv.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|data_csv|string, required|Dataset Param: Data csv path.|
|data_shape|Shape(tuple), required|Dataset Param: Shape of the data.|
|label_csv|string, optional, default='NULL'|Dataset Param: Label csv path. If is NULL, all labels will be returned as 0|
|label_shape|Shape(tuple), optional, default=(1,)|Dataset Param: Shape of the label.|

* Name: `ImageRecordIter`
* Description: Create iterator for dataset packed in recordio.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|path_imglist|string, optional, default=''|Dataset Param: Path to image list.|
|path_imgrec|string, optional, default='./data/imgrec.rec'|Dataset Param: Path to image record file.|
|aug_seq|string, optional, default='aug_default'|Augmentation Param: the augmenter names to represent sequence of augmenters to be applied, seperated by comma. Additional keyword parameters will be seen by these augmenters.|
|label_width|int, optional, default='1'|Dataset Param: How many labels for an image.|
|data_shape|Shape(tuple), required|Dataset Param: Shape of each instance generated by the DataIter.|
|preprocess_threads|int, optional, default='4'|Backend Param: Number of thread to do preprocessing.|
|verbose|boolean, optional, default=True|Auxiliary Param: Whether to output parser information.|
|num_parts|int, optional, default='1'|partition the data into multiple parts|
|part_index|int, optional, default='0'|the index of the part will read|
|shuffle|boolean, optional, default=False|Augmentation Param: Whether to shuffle data.|
|seed|int, optional, default='0'|Augmentation Param: Random Seed.|
|verbose|boolean, optional, default=True|Auxiliary Param: Whether to output information.|
|batch_size|int (non-negative), required|Batch Param: Batch size.|
|round_batch|boolean, optional, default=True|Batch Param: Use round robin to handle overflow batch.|
|prefetch_buffer|, optional, default=4|Backend Param: Number of prefetched parameters|
|rand_crop|boolean, optional, default=False|Augmentation Param: Whether to random crop on the image|
|crop_y_start|int, optional, default='-1'|Augmentation Param: Where to nonrandom crop on y.|
|crop_x_start|int, optional, default='-1'|Augmentation Param: Where to nonrandom crop on x.|
|max_rotate_angle|int, optional, default='0'|Augmentation Param: rotated randomly in [-max_rotate_angle, max_rotate_angle].|
|max_aspect_ratio|float, optional, default=0|Augmentation Param: denotes the max ratio of random aspect ratio augmentation.|
|max_shear_ratio|float, optional, default=0|Augmentation Param: denotes the max random shearing ratio.|
|max_crop_size|int, optional, default='-1'|Augmentation Param: Maximum crop size.|
|min_crop_size|int, optional, default='-1'|Augmentation Param: Minimum crop size.|
|max_random_scale|float, optional, default=1|Augmentation Param: Maxmum scale ratio.|
|min_random_scale|float, optional, default=1|Augmentation Param: Minimum scale ratio.|
|max_img_size|float, optional, default=1e+10|Augmentation Param: Maxmum image size after resizing.|
|min_img_size|float, optional, default=0|Augmentation Param: Minimum image size after resizing.|
|random_h|int, optional, default='0'|Augmentation Param: Maximum value of H channel in HSL color space.|
|random_s|int, optional, default='0'|Augmentation Param: Maximum value of S channel in HSL color space.|
|random_l|int, optional, default='0'|Augmentation Param: Maximum value of L channel in HSL color space.|
|rotate|int, optional, default='-1'|Augmentation Param: Rotate angle.|
|fill_value|int, optional, default='255'|Augmentation Param: Maximum value of illumination variation.|
|data_shape|Shape(tuple), required|Dataset Param: Shape of each instance generated by the DataIter.|
|inter_method|int, optional, default='1'|Augmentation Param: 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand.|
|pad|int, optional, default='0'|Augmentation Param: Padding size.|
|seed|int, optional, default='0'|Augmentation Param: Random Seed.|
|mirror|boolean, optional, default=False|Augmentation Param: Whether to mirror the image.|
|rand_mirror|boolean, optional, default=False|Augmentation Param: Whether to mirror the image randomly.|
|mean_img|string, optional, default=''|Augmentation Param: Mean Image to be subtracted.|
|mean_r|float, optional, default=0|Augmentation Param: Mean value on R channel.|
|mean_g|float, optional, default=0|Augmentation Param: Mean value on G channel.|
|mean_b|float, optional, default=0|Augmentation Param: Mean value on B channel.|
|mean_a|float, optional, default=0|Augmentation Param: Mean value on Alpha channel.|
|scale|float, optional, default=1|Augmentation Param: Scale in color space.|
|max_random_contrast|float, optional, default=0|Augmentation Param: Maximum ratio of contrast variation.|
|max_random_illumination|float, optional, default=0|Augmentation Param: Maximum value of illumination variation.|
|verbose|boolean, optional, default=True|Augmentation Param: Whether to print augmentor info.|

* Name: `MNISTIter`
* Description: Create iterator for MNIST hand-written digit number recognition dataset.
* Arguments: 

| Name | Type info | Description |
| --- | --- | --- |
|image|string, optional, default='./train-images-idx3-ubyte'|Dataset Param: Mnist image path.|
|label|string, optional, default='./train-labels-idx1-ubyte'|Dataset Param: Mnist label path.|
|batch_size|int, optional, default='128'|Batch Param: Batch Size.|
|shuffle|boolean, optional, default=True|Augmentation Param: Whether to shuffle data.|
|flat|boolean, optional, default=False|Augmentation Param: Whether to flat the data into 1D.|
|seed|int, optional, default='0'|Augmentation Param: Random Seed.|
|silent|boolean, optional, default=False|Auxiliary Param: Whether to print out data info.|
|num_parts|int, optional, default='1'|partition the data into multiple parts|
|part_index|int, optional, default='0'|the index of the part will read|
|prefetch_buffer|, optional, default=4|Backend Param: Number of prefetched parameters|

