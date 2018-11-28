import tensorflow as tf
import numpy as np
import pickle
import platform

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

trainDataFilePath = "D:\\pycharm_program\\DATA\\cifar-10-batches-py\\data_batch_1"
valDataFilePath = "D:\\pycharm_program\\DATA\\cifar-10-batches-py\\test_batch"


class Cifar10Data(object):
    def __init__(self, mode, batch_size, buffer_size=1000):
        """Create the cifar-10-data"""
        self.dataSize = 10000
        self.num_classes = 10

        # retrieve the data from the cifar-10 file
        self._read_cifar10_file(mode)

        # shuffle the data and get the batch_size images
        self._shuffle_lists(batch_size)
        # convert the list to TF
        self.imgs = convert_to_tensor(self.imgs, dtype=dtypes.int8)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.imgs, self.labels))


        # convert the dataset to adapt the AlextNet
        if mode == 'training':
            data = data.map(self._parse_function_train)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))
        self.data = data



    def _read_cifar10_file(self, mode):
        if mode == "training":
            with open(trainDataFilePath, 'rb') as fo:
                datadict = self._load_pickle(fo)
                self.imgs = datadict['data'].tolist()
                self.labels = datadict['labels']
        else:
            with open(valDataFilePath, 'rb') as fo:
                datadict = pickle.load(fo)
                self.imgs = datadict['data'].tolist()
                self.labels = datadict['labels']

    def _shuffle_lists(self, batch_size):
        """Conjoined shuffling of the list of paths and labels."""
        imgs = self.imgs
        labels = self.labels
        permutation = np.random.permutation(self.dataSize)
        self.imgs = []
        self.labels = []
        start = 1
        for i in permutation:
            if start == batch_size:
                break
            else:
                self.imgs.append(imgs[i])
                self.labels.append(labels[i])
                start = start + 1

    def _parse_function_train(self, img, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img = tf.reshape(img, [32, 32, 3])
        img_resized = tf.image.resize_images(img, [227, 227])

        # RGB -> BGR
        img_bgr = img_resized[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_inference(self, img, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img = tf.reshape(img, [32, 32, 3])
        img_resized = tf.image.resize_images(img, [227, 227])


        # RGB -> BGR
        img_bgr = img_resized[:, :, ::-1]

        return img_bgr, one_hot

    # 读取文件
    def _load_pickle(self, fo):
        version = platform.python_version_tuple()  # 取python版本号
        if version[0] == '2':
            return pickle.load(fo)  # pickle.load, 反序列化为python的数据类型
        elif version[0] == '3':
            return pickle.load(fo, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))
# test
if __name__ == "__main__":
    test = Cifar10Data("training", 10)
