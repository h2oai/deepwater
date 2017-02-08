import unittest

from deepwater.models import alexnet
from deepwater import optimizers

from deepwater.models.test_utils import LARGE_must_converge
from deepwater.models.test_utils import MNIST_must_converge


class TestAlexnet(unittest.TestCase):
    def test_alexnet_must_converge_on_MNIST(self):
        MNIST_must_converge("alexnet", alexnet.AlexNet,
                            optimizers.RMSPropOptimizer,
                            batch_size=32,
                            epochs=3,
                            initial_learning_rate=0.2,
                            summaries=False,
                            use_debug_session=False)

    def test_alexnet_large_must_converge(self):
        LARGE_must_converge("alexnet", alexnet.AlexNet,
                            optimizers.RMSPropOptimizer,
                            batch_size=32,
                            epochs=1,
                            initial_learning_rate=0.2,
                            summaries=True)


if __name__ == "__main__":
    unittest.main()
