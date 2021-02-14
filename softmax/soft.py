# import numpy as np
# import tensorflow.compat.v1 as tf
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# tf.disable_v2_behavior()
# tf.disable_eager_execution()
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow_datasets as tfds
#
#
# mnist = tfds.load(name = "mnist", one_hot=True)
#
# x = tf.placeholder(tf.float32, [None, 784])
# y_ = tf.placeholder(tf.float32, [None, 10])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100) #MB-GD
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))