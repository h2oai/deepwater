import unittest

from deepwater.models import resnet
from deepwater import optimizers

from deepwater.models.test_utils import CIFAR10_must_converge
from deepwater.models.test_utils import MNIST_must_converge

import tensorflow

class TestResnet(unittest.TestCase):

    def test_resnet_must_converge_on_MNIST(self):
        MNIST_must_converge("Resnet", resnet.ResNet,
                            optimizers.RMSPropOptimizer,
                            batch_size=32,
                            epochs=10,
                            initial_learning_rate=0.001,
                            )


#     def test_resnet_must_converge_on_CIFAR10(self):
#         CIFAR10_must_converge("Resnet", resnet.Resnet,
#                               optimizers.RMSPropOptimizer,
#                               batch_size=16,
#                               epochs=90,
#                               initial_learning_rate=0.001,
#                               summaries=False,
#                               use_debug_session=False,
#                               )

if __name__ == "__main__":
    unittest.main()
