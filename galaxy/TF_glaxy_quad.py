# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
import os
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':

    tf.disable_v2_behavior()
    tf.disable_eager_execution()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    features = 2
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.zeros([features, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    data_x = np.array([[2, 4], [3, 9], [4, 16], [6, 36], [7, 49]])
    data_y = np.array([[70], [110], [165], [390], [550]])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(0, 100000):
        sess.run(update, feed_dict={x: data_x, y_: data_y})
        if i % 10000 == 0:
            print('Iteration:', i, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:',
                  loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}))
    print('Prediction for Galaxy S5:', np.matmul(np.array([5, 25]), sess.run(W)) + sess.run(b))
    print('Prediction for Galaxy S1:', np.matmul(np.array([1, 1]), sess.run(W)) + sess.run(b))


