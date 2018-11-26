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
        :param txt_file:
        :param mode:
        :param batch_size:
        :param num_classes:
        :param shuffle:
        :param buffer_size:
        """
