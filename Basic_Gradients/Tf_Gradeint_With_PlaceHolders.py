# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    # Q : What's the difference between tf.placeholder and tf.Variable?
    # https://stackoverflow.com/questions/36693740/whats-the-difference-between-tf-placeholder-and-tf-variable
    # ANS : use tf.Variable for trainable variables
    # such as weights (W) and biases (B) for your model. tf.Variable you have to provide an initial value when you
    # declare it. With tf.placeholder you don't have to provide an initial value and you can specify it at run time
    # with the feed_dict.

    x = tf.placeholder(tf.float32, [None, 1])
    y_ = tf.placeholder(tf.float32, [None, 1])

    w = tf.Variable(0.)  # note the dot
    y = w * x
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for _ in range(0, 1000):
        sess.run(update, feed_dict={x: [[7.01], [3.02], [4.99], [8.]], y_: [[14.01], [6.01], [10.], [16.04]]})
    print(sess.run(w))
