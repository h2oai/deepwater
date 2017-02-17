import unittest

from deepwater.models import vgg
from deepwater import optimizers

from deepwater.models.test_utils import CIFAR10_must_converge
from deepwater.models.test_utils import MNIST_must_converge
from deepwater.models.test_utils import cat_dog_mouse_must_converge


class TestVGG(unittest.TestCase):

    def xxx_test_vgg_must_converge_on_CIFAR10(self):
        CIFAR10_must_converge("vgg16", vgg.VGG16,
                            optimizers.MomentumOptimizer,
                            batch_size=16,
                            epochs=3,
                            initial_learning_rate=0.2,
                            summaries=False,
                            use_debug_session=False,
                            )

    def test_vgg_must_converge_on_MNIST(self):
        MNIST_must_converge("vgg16", vgg.VGG16,
                              optimizers.MomentumOptimizer,
                              batch_size=16,
                              epochs=3,
                              initial_learning_rate=1e-5,
                              summaries=False,
                              use_debug_session=False,
                            )

    def test_vgg_cat_dog_mouse_must_converge(self):
        train_error = cat_dog_mouse_must_converge("vgg16", vgg.VGG16,
                                                  optimizers.MomentumOptimizer,
                                                  batch_size=32,
                                                  epochs=50,
                                                  initial_learning_rate=1e-5,
                                                  summaries=False)
        self.assertTrue(train_error <= 0.1)

if __name__ == "__main__":
    unittest.main()
