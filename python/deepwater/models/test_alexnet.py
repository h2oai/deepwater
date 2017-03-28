import unittest

from deepwater.models import alexnet
from deepwater import optimizers

from deepwater.models.test_utils import cat_dog_mouse_must_converge
from deepwater.models.test_utils import MNIST_must_converge


class TestAlexnet(unittest.TestCase):
    #
    # def test_alexnet_MNIST_must_converge(self):
    #     MNIST_must_converge("alexnet", alexnet.AlexNet,
    #                         optimizers.MomentumOptimizer,
    #                         batch_size=32,
    #                         epochs=5,
    #                         initial_learning_rate=1e-2,
    #                         summaries=False,
    #                         use_debug_session=False)

    def test_alexnet_cat_dog_mouse_must_converge(self):
        train_error = cat_dog_mouse_must_converge("alexnet", alexnet.AlexNet,
                            optimizers.AdamOptimizer,
                            batch_size=32,
                            epochs=80,
                            initial_learning_rate=1e-4,
                            summaries=True)
        self.assertTrue(train_error <= 0.1)


if __name__ == "__main__":
    unittest.main()
