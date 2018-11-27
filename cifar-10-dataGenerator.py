import tensorflow as tf
import numpy as np
import pickle

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

trainDataFilePath = "D:\\pycharm_program\\DATA\\cifar-10-batches-py\\data_batch_1"
valDataFilePath = "D:\\pycharm_program\\DATA\\cifar-10-batches-py\\test_batch"

class Cifar10Data(object):
    def __init__(self, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):
        """Create the cifar-10-data"""
        self.dataSize = 10000
        self.num_classes = 10

        # retrieve the data from the cifar-10 file
        self._read_cifar10_file(mode)

        # shuffle the data
        if shuffle:
            self._shuffle_lists()
        # convert the list to TF
        self.imgs = convert_to_tensor(self.imgs, dtype=dtypes.int8)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)




    def _read_cifar10_file(self, mode):
        if mode == "training":
            with open(trainDataFilePath, 'rb') as fo:
                datadict = pickle.load(fo, encoding='latin1')
                self.imgs = datadict['data'].tolist()
                self.labels = datadict['labels']
        else:
            with open(valDataFilePath, 'rb') as fo:
                datadict = pickle.load(fo, encoding='latin1')
                self.imgs = datadict['data'].tolist()
                self.labels = datadict['labels']

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        imgs = self.imgs
        labels = self.labels
        permutation = np.random.permutation(self.dataSize)
        self.imgs = []
        self.labels = []
        for i in permutation:
            self.imgs.append(imgs[i])
            self.labels.append(labels[i])

