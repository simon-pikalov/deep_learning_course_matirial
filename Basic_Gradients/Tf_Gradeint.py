# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    x = tf.constant([7.01, 3.02, 4.99, 8.])
    y_ = tf.constant([14.01, 6.01, 10., 16.04])
    w = tf.Variable(0.)  # note the dot
    y = w * x
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for _ in range(0, 1000):
        sess.run(update)
    print(sess.run(w))