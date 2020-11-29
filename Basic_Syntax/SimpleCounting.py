# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    x = tf.Variable(1)
    step = tf.constant(2)
    update = tf.assign(x, x + step)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(4):
        print(sess.run(update))
    print(sess.run(x))
    print(sess.run(x))
    sess.run(tf.global_variables_initializer())
    print(sess.run(x))
