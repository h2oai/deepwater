from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import tarfile

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import cPickle

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import numpy as np


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
      f: A file object that can be passed into a gzip reader.
    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
      ValueError: If the bytestream does not start with 2051.
    """
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    print('Extracting', f.name)
    with tarfile.TarFile(fileobj=gzip.GzipFile(fileobj=f)) as tfile:
        for name in tfile.getnames():
            if 'data_batch' in name:
                fd = tfile.extractfile(name)
                data = cPickle.load(fd)
                for image, label in zip(data['data'], data['labels']):
                    image = np.array(image)
                    train_images.append(image)
                    train_labels.append(label)
            elif 'test_batch' in name:
                fd = tfile.extractfile(name)
                data = cPickle.load(fd)
                for image, label in zip(data['data'], data['labels']):
                    image = np.array(image)
                    test_images.append(image)
                    test_labels.append(label)
            else:
                continue

    return train_images, train_labels, test_images, test_labels


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 # one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True):
        # """Construct a DataSet.
        # one_hot arg is used only if fake_data is true.  `dtype` can be either
        # `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        # `[0, 1]`.
        # """
        # dtype = dtypes.as_dtype(dtype).base_dtype
        # if dtype not in (dtypes.uint8, dtypes.float32):
        #   raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
        #                   dtype)
        # if fake_data:
        #   self._num_examples = 10000
        #   self.one_hot = one_hot
        # else:
        assert len(images) == len(labels), "images len != labels len"
        self._num_examples = len(images)

        #   # Convert shape from [num examples, rows, columns, depth]
        #   # to [num examples, rows*columns] (assuming depth == 1)
        #   if reshape:
        #     assert images.shape[3] == 1
        #     images = images.reshape(images.shape[0],
        #                             images.shape[1] * images.shape[2])
        #   if dtype == dtypes.float32:
        #     # Convert from [0, 255] -> [0.0, 1.0].
        #     images = images.astype(numpy.float32)
        #     images = numpy.multiply(images, 1.0 / 255.0)
        self._images = np.array(images)
        self._labels = np.array(labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000):

    TRAIN_TEST_IMAGES = 'cifar-10-python.tar.gz'
    SOURCE_TRAIN_TEST = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    local_file = base.maybe_download(TRAIN_TEST_IMAGES, train_dir,
                                     SOURCE_TRAIN_TEST)

    with open(local_file, 'rb') as f:
        train_images, train_labels, test_images, test_labels = extract_images(
            f)

    # local_file = base.maybe_download(TRAIN_LABELS, train_dir,
    #                                  SOURCE_URL + TRAIN_LABELS)
    # with open(local_file, 'rb') as f:
    #   train_labels = extract_labels(f, one_hot=one_hot)

    # local_file = base.maybe_download(TEST_IMAGES, train_dir,
    #                                  SOURCE_URL + TEST_IMAGES)
    # with open(local_file, 'rb') as f:
    #   test_images = extract_images(f)

    # local_file = base.maybe_download(TEST_LABELS, train_dir,
    #                                  SOURCE_URL + TEST_LABELS)
    # with open(local_file, 'rb') as f:
    #   test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images,
                         validation_labels,
                         dtype=dtype,
                         reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    return base.Datasets(train=train, validation=validation, test=test)


def load_cifar10(train_dir='CIFAR10-data'):
    return read_data_sets(train_dir)


if __name__ == "__main__":
    load_cifar10()
