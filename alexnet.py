import tensorflow as tf
import numpy as np

class AlexNet(object):
    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path="DEFAULT"):

        # Parse input arguments into class variable
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == "DEFAULT":
            self.WEIGHTS_PATH = "bvlc_alexnet.npy"
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        # first layer: conv(w, relu) -> pool ->lrn
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')

        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # second layer :conv(w, relu) -> lrn->pool with 2 grounds
        conv2= conv(pool1, 5, 5,  256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # third layer: conv(w, relu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # fourth layer: conv(w, Relu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # fifth layer: conv(w, relu)->pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # sixth layer: Flatten->FC(w, Relu)->Dropout
        flattened = tf.reshape(pool5, [6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # seventh layer: FC(w,Relu)->Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # eighth layer: FC and return unscaled activations
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    def load_initial_weights(self, session):
        """
         load weights from file into network


        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists(e.g.weights['conv1'] is a list) and not as dict of dicts
        (e.g. weights['conv1'] is a dict with keys 'weights'&'biase') we need a special load function
        """
        # load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # loop over al layer names stored in the weights dict
        for op_name in weights_dict:

            # check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):
                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        #weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))













"""
Predefine all neccssary layer for the AlexNet
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:          # ??
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])

        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        # https://blog.csdn.net/hhy_csdn/article/details/80030468(群卷积详解)
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weights_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)

        output_groups = [convolve(i, k) for i, k in zip(input_groups, weights_groups)]

        # Concat the convolved ouput together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """
    Create a local response normalization layer
    """
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):

    """Create a max pooling layer"""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

def fc(x, num_in, num_out, name, relu=True):
    """ Create a fully connected layer"""

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights an biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    if relu:

        # apply Relu non linearity
        relu = tf.nn.relu(act)
    else:
        return act

def dropout(x, keep_prob):
    """ Create a dropout layer"""
    return tf.nn.dropout(x, keep_prob)








