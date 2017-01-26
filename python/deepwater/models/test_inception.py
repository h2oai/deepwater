import unittest

from deepwater.models import inception
from deepwater import optimizers

from deepwater.models.test_utils import CIFAR10_must_converge
from deepwater.models.test_utils import MNIST_must_converge


class TestInceptionResnetV2(unittest.TestCase):
    pass

    # def test_inceptionv4_must_converge_on_MNIST(self):
    #     MNIST_must_converge("inceptionResnetV2", inception.InceptionResNetV2,
    #                         optimizers.RMSPropOptimizer,
    #                         batch_size=32,
    #                         epochs=90,
    #                         initial_learning_rate=0.001,
    #                         summaries=False,
    #                         use_debug_session=False,
    #                         )


#     def test_inceptionv4_must_converge_on_CIFAR10(self):
#         CIFAR10_must_converge("inceptionv4", inception.InceptionV4,
#                               optimizers.RMSPropOptimizer,
#                               batch_size=16,
#                               epochs=90,
#                               initial_learning_rate=0.001,
#                               summaries=False,
#                               use_debug_session=False,
#                               )

class TestInceptionV4(unittest.TestCase):

    def test_inceptionv4_must_converge_on_MNIST(self):
        MNIST_must_converge("inceptionV4", inception.InceptionV4,
                            optimizers.RMSPropOptimizer,
                            batch_size=32,
                            epochs=90,
                            initial_learning_rate=0.2,
                            summaries=False,
                            use_debug_session=False,
                            )

    def test_inceptionv4_must_converge_on_CIFAR10(self):
        CIFAR10_must_converge("inceptionv4", inception.InceptionV4,
                              optimizers.RMSPropOptimizer,
                              batch_size=16,
                              epochs=90,
                              initial_learning_rate=0.2,
                              summaries=False,
                              use_debug_session=False,
                              )


if __name__ == "__main__":
    unittest.main()
