"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf
import cifar10dataGenerator

from alexnet import AlexNet
from datetime import datetime


"""
Configuration Part.
"""
"""
获取当前地址
"""
# 更改工作目录
abspath = os.path.abspath(__file__)  # 获取当前文件绝对地址
# E:\GitHub\TF_Cookbook\08_Convolutional_Neural_Networks\03_CNN_CIFAR10\ostest.py
dname = os.path.dirname(abspath)  # 获取文件所在文件夹地址
# E:\GitHub\TF_Cookbook\08_Convolutional_Neural_Networks\03_CNN_CIFAR10
os.chdir(dname)  # 转换目录文件夹到上层



# Learning params
learning_rate = 0.001
num_epochs = 1
batch_size = 100

# Network params
dropout_rate = 0.5
num_classes = 10
train_layers = ['fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = dname + "\\tmp\\tensorboard"
checkpoint_path = dname + "\\tmp\\checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    # 数据集cifar-10的文件夹位置
    data_dir = dname + "\\temp\\cifar-10-batches-bin"
    tr_imgs, tr_labels  = cifar10dataGenerator.inputs(False, data_dir, batch_size)
    val_imgs, val_labels = cifar10dataGenerator.inputs(True, data_dir, batch_size)



# 开始时先训练
x = tr_imgs
y = tr_labels
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(50000/batch_size))
val_batches_per_epoch = int(np.floor(10000/batch_size))

# Start Tensorflow session
with tf.Session() as sess:
    # 获得协调对象
    coord = tf.train.Coordinator()
    sess.run(tf.local_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)
    saver.restore(sess, checkpoint_path +"\\model_epoch1.ckpt")

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):
        x = tr_imgs
        y = tr_labels
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        for step in range(train_batches_per_epoch):

            # And run the training op
            sess.run(train_op, feed_dict={keep_prob: dropout_rate})

            # 标志
            print(step + 1)
            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        x = val_imgs
        y = val_labels
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            acc = sess.run(accuracy, feed_dict={keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
    coord.request_stop()  # 请求线程结束
    coord.join()  # 等待线程结束