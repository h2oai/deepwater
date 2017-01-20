import tensorflow as tf

from deepwater.models import BaseImageClassificationModel

from deepwater.models.nn import block, fc, max_pool_2x2


class VGG16(BaseImageClassificationModel):

    def __init__(self, width=28, height=28, channels=1, classes=10,
                 hidden_layers=[],
                 dropout=[]):
        super(VGG16, self).__init__()

        assert width == height, "width and height must be the same"

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")
        self._inputs = x

        x = tf.reshape(x, [-1, width, height, channels])

        self._number_of_classes = classes
    
        if width < 224:
            x = tf.image.resize_images(x, [48, 48])
            input_width = 48 
        elif width == 224:
            input_width = 224 
        else:
            x = tf.image.resize_images(x, [224, 224])
            input_width = 224 

        # 2 x 64
        out = block(x, [3, 3, channels, 64])
        out = block(out, [3, 3, 64, 64])
        out = max_pool_2x2(out)
        input_width /= 2

        # 2 x 128
        out = block(out, [3, 3, 64, 128])
        out = block(out, [3, 3, 128, 128])
        out = max_pool_2x2(out)
        input_width /= 2

        # 3 x 256
        out = block(out, [3, 3, 128, 256])
        out = block(out, [3, 3, 256, 256])
        out = block(out, [3, 3, 256, 256])
        out = max_pool_2x2(out)
        input_width /= 2

        # 3 x 512
        out = block(out, [3, 3, 256, 512])
        out = block(out, [3, 3, 512, 512])
        out = block(out, [3, 3, 512, 512])
        out = max_pool_2x2(out)
        input_width /= 2

        # 512
        out = block(out, [3, 3, 512, 512])
        out = block(out, [3, 3, 512, 512])
        out = block(out, [3, 3, 512, 512])
        out = max_pool_2x2(out)
        input_width /= 2

        flatten_size = reduce(lambda a,b: a * b, out.get_shape().as_list()[1:], 1)

        out = tf.reshape(out, [-1, int(flatten_size)])

        # fully connected
        out = fc(out, [int(flatten_size), 4096])
        out = tf.nn.relu(out)
        out = fc(out, [4096, 4096])
        out = tf.nn.relu(out)
        y = fc(out, [4096, classes])

        self._logits = y

        if classes > 1:
            self._predictions = tf.nn.softmax(y)
        else:
            self._predictions = self._logits

    @property
    def train_dict(self):
        return {}

    @property
    def name(self):
        return "vgg"

    @property
    def number_of_classes(self):
        return self._number_of_classes

    @property
    def inputs(self):
        return self._inputs

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions
