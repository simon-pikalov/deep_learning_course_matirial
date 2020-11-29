# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
import  os
tf.disable_v2_behavior()
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':

    print("\n\n\n\n\n\n\n*********************************\n\n\n\n\n\n\n")
    print("Program starts here\n")
    var1 = tf.Variable(3)
    var2 = tf.Variable(4)
    print("var 1 :",var1 , ", var 2 ", var2)
    c2 = var1 * var2
    print(c2)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ans_v1=sess.run(var1)
    ans_c2 = sess.run(c2)
    print("ans_v1  :",ans_v1 , ", ans_c2 ", ans_c2)
    print("Program ends here")