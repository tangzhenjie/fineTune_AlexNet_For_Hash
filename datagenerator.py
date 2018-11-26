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



