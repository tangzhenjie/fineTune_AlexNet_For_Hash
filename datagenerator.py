"""containes a helper class for image input piplines in tensorflow"""

import tensorflow as tf
import numpy as np

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

class ImageDataGenerator(object):
    """
    wrapper class around the new tensorflow dataset pipline

    Require Tensorflow >= version1.12rc0
    """
    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):
        """
        create a new ImageDataGenerator

        Recieves a path string to a text file, which consist of many lines,
        where each line has first a path to an image and seperated  by
        a space an integer, referring to the class number.Using this data,
        this class will create Tensorflow datasets, that can be used to train

        :param txt_file: path to the txt file
        :param mode: Either 'training' or 'validation'. Depending on this value,
        different parsing function will be used
        :param batch_size: Number of images per batch
        :param num_classes: Number of classes in the datasets
        :param shuffle: wether or not to shuffle the data in the dataset and the initial file list
        :param buffer_size: Number of images used as buffer for Tensorflows shuffling of the dataset

        Raise:
            ValueError: if an invalid mode is passed
        """
        self.txt_file = txt_file
        self.num_class = num_classes

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffing(打乱) of the file and label list(together)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtypes=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == "training":
            data = data.map(self._parse_function_train, num_threads=8,
                            output_buffer_size=100*batch_size)
        elif mode == "inference":
            data = data.map(self._parse_function_inference, num_threads=8,
                            output_buffer_size=100*batch_size)
        else:
            raise ValueError("Invalid mode '%s'." %(mode))





        def _read_txt_file(self):
            """Read the content of the text file and store it into list
            """
            self.img_paths = []
            self.labels = []

            with open(self.txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split(" ")
                    self.img_paths.append(items[0])
                    self.labels.append(int(items[1]))

        def _shuffle_lists(self):
            """Conjoined shuffling of the list of paths and labels"""
            path = self.img_paths
            labels = self.labels
            permutation = np.random.permutation(self.data_size)
            self.img_paths = []
            self.labels = []
            for i in permutation:
                self.img_paths.append(path[i])
                self.labels.append(labels[i])

        def _parse_function_train(self, filename, label):

            """Input parser for samples of the training set"""

            # convert label number into one-hot-encoding
            one_hot = tf.one_hot(label, self.num_classes)# from the o

            # load an preprocess the image
            img_string = tf.read_file(filename)
            img_decoded = tf.image.decode_png(img_string, channels=3)
            img_resized = tf.image.resize_images(img_decoded, [227, 227])

            # Dataaugmentation(数据扩充) comes here
            img_centered = tf.subtract(img_resized, IMAGENET_MEAN) # ???

            # RGB -> BGR
            img_bgr = img_centered[:, :, ::-1]

            return img_bgr, one_hot

        def _parse_function_inference(self, filename, label):
            """Input parser for samples of the of validation/test set"""

            # convert label number into one-hot-encoding
            one_hot = tf.one_hot(label, self.num_classes)

            # load and preprocess the image
            img_string = tf.read_file(filename)
            img_decoded = tf.image.decode_png(img_string, channels=3)
            img_resized = tf.image.resize_images(img_decoded, [227, 227])
            img_centered = tf.subtract(img_resized,IMAGENET_MEAN)

            # RGB -> BGR
            img_bgr = img_centered[:, :, ::-1]
            return img_bgr, one_hot

