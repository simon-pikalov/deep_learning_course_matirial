import tensorflow.compat.v1 as tf
import  os
tf.disable_v2_behavior()
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("\n\n\n\n\n\n\n*********************************\n\n\n\n\n\n\n")
print("Program starts here\n")


if __name__ == '__main__':

    a = tf.constant(3)
    print("a",a)
    b = tf.constant(4)
    c = a*b
    print("c",c)
    sess = tf.Session()
    print("run a ",sess.run(a))
    print("run c ",sess.run(c))
