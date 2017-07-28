import unittest

from deepwater.models import lenet
from deepwater import optimizers

from deepwater.models.test_utils import MNIST_must_converge, cat_dog_mouse_must_converge

# from deepwater.models.nn import get_avaialble_gpus

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class TestLenet(unittest.TestCase):
    # gpus = 4
    avail_gpus = get_available_gpus()
    gpus = avail_gpus
    print("available gpus %d\n" % (len(avail_gpus)))

    def test_lenet(self):
        MNIST_must_converge('lenet', lenet.LeNet,
                            optimizers.AdamOptimizer,
                            # optimizers.MomentumOptimizer,
                            initial_learning_rate=1e-3,
                            batch_size=32,
                            epochs=5)

    def test_lenet_cat_dog_mouse_must_converge_28(self):
        train_error = cat_dog_mouse_must_converge("lenet", lenet.LeNet,
                                                  optimizers.AdamOptimizer,
                                                  # optimizers.MomentumOptimizer,
                                                  batch_size=32 * TestLenet.gpus,
                                                  epochs=80,
                                                  initial_learning_rate=1e-3,
                                                  summaries=True,
                                                  dim=28)
        self.assertTrue(train_error <= 0.1)

    def test_lenet_cat_dog_mouse_must_converge_299(self):
        train_error = cat_dog_mouse_must_converge("lenet", lenet.LeNet,
                                                  optimizers.AdamOptimizer,
                                                  # optimizers.MomentumOptimizer,
                                                  batch_size=16,  # *TestLenet.gpus,
                                                  epochs=25,
                                                  initial_learning_rate=1e-4,  # rate for new fc
                                                  summaries=True,
                                                  dim=299)
        self.assertTrue(train_error <= 0.1)


if __name__ == "__main__":
    unittest.main()
