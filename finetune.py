"""Script to finetune AlexNet using tensorflow"""

import os
import numpy as np
import tensorflow as tf
import cifar10_input

from alexnet import AlexNet
from datetime import datetime

"""
Configuration Part.
"""

# 学习参数
learning_rate = 0.0009     # 学习率
num_epochs = 10            # 学习轮数（每一轮在所有训练集上训练一次）
batch_size = 10            # 批量大小

# 网络参数
dropout_rate = 0.5         # 保留的概率
num_classes = 10           # 网络输出的种类数
train_layers = ['fc8', 'fc7', 'fc6']   # 需要训练的网络层

# 多久写tf.summary数据到磁盘
display_step = 20

# 写入文件地址
filewriter_path = "D:\\pycharm_program\\finetune_alexnet\\tmp\\tensorboard"
checkpoint_path = "D:\\pycharm_program\\finetune_alexnet\\tmp\\checkpoints"

# 如果不存在就新建一个
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

"""
以下建立图的过程和部分变量
"""
# 获取数据通道pipline(训练数据)
train_imgs, train_labels = cifar10_input.input_pipeline(batch_size, train_logical=True)
val_imgs, val_labels = cifar10_input.input_pipeline(batch_size, train_logical=False)

# 初始化(因为先训练)
x = train_imgs
y = train_labels


# 定义图的输入dropout_rate的占位符
keep_prob = tf.placeholder(tf.float32)

# 初始化模型
model = AlexNet(x, keep_prob, num_classes, train_layers)

# 关联上网络的输出层
score = model.fc8

# 列出需要训练的层scope
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# 定义损失函数层
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# 定义训练操作
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# 添加梯度到summary中（tensorboard）
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# 添加变量到summary中（tensorboard）
for var in var_list:
    tf.summary.histogram(var.name, var)

# 添加损失到summary中（tensorboard）
tf.summary.scalar('cross_entropy', loss)


# 定义评价操作层
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 添加准确率到summary（TensorBoard）
tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged_summary = tf.summary.merge_all()

# 初始化写入summary的文件地址
writer = tf.summary.FileWriter(filewriter_path)

# 初始化一个saver对象用来保存和恢复图中的variable
saver = tf.train.Saver()




# 根据批量大小来获取一个epoch需要多少个的批量（因为一个epoch需要训练或者评价一次所有的样本）
# ：这里写死了因为cifar-10有50000个训练图片和10000测试图片
train_batches_per_epoch = int(np.floor(50000/batch_size))
val_batches_per_epoch = int(np.floor(10000/batch_size))

"""
以上建立图的过程和部分变量
"""

# 开始执行上面构建好的graph
with tf.Session() as sess:

    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # 添加图graph到TensorBoard
    writer.add_graph(sess.graph)

    # 加载不训练层的参数
    model.load_initial_weights(sess)

    #saver.restore(sess, "D:\\pycharm_program\\finetune_alexnet\\tmp\\checkpoints\\model_epoch10.ckpt")

    # 输出开始训练提示信息
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # 获得协调对象
    coord = tf.train.Coordinator()
    # 开启队列
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 循环epochs
    for epoch in range(num_epochs):
        # 改变成训练数据pipline
        x = train_imgs
        y = train_labels
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        for step in range(train_batches_per_epoch):

            # 训练网络
            sess.run(train_op, feed_dict={keep_prob: dropout_rate})

            # 用当前的数据生成summary然后写入文件
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={keep_prob: dropout_rate})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)
        # 改变为训练pipline
        x = val_imgs
        y = val_labels

        # 开始在整个验证集上验证
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
        # 保存checkpoint of the model
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
        coord.request_stop()  # 请求线程结束
        coord.join()  # 等待线程结束