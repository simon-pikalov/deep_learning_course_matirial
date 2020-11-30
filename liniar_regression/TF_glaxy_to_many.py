# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
import os
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    features = 20
    suff_feat = np.arange(1, features + 1)
    np.random.shuffle(suff_feat)


    def vecto(x):
        ret = []
        for i in suff_feat:
            ret.append(x ** i / 7. ** i)
        return ret


    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.zeros([features, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    data_x = np.array([vecto(2), vecto(3), vecto(4), vecto(6), vecto(7)])
    data_y = np.array([[70], [110], [165], [390], [550]])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(0, 10000):
        sess.run(update, feed_dict={x: data_x, y_: data_y})
        if i % 10000 == 0:
            print('Iteration:', i, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:',
                  loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}))
    x_axis = np.arange(0, 8, 0.1)
    x_data = []
    for i in x_axis:
        x_data.append(vecto(i))
    x_data = np.array(x_data)
    y_vals = np.matmul(x_data, sess.run(W)) + sess.run(b)
    plt.plot(x_axis, y_vals,data_y)
    plt.show()