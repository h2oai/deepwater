import tensorflow as tf

if "concat_v2" in dir(tf):
    def concat(axis, tensors, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(axis, tensors, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)