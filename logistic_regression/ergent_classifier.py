import numpy as np
import tensorflow.compat.v1 as tf
import os
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


vocabulary_size = 0  # can use "global" keyword
word2location = {}
data = ["Where are you? I'm trying to reach you for half an hour already, contact me ASAP I need to leave now!",
        "I want to go out for lunch, let me know in the next couple of minutes if you would like to join.",
        "I was wondering whether you are planning to  attend the party we are having next month.",
        "I wanted to share my thoughts with you."]


def prepare_vocabulary(data):
    idx = 0
    for sentence in data:
        for word in sentence.split():  # better use nltk.word_tokenize(sentence) and perform some stemming etc.!!!
            if word not in word2location:
                word2location[word] = idx
                idx += 1
    return idx


def convert2vec(sentence):
    res_vec = np.zeros(vocabulary_size)
    for word in sentence.split():  # also here...
        if word in word2location:
            res_vec[word2location[word]] += 1
    return res_vec

def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))

features = vocabulary_size
eps = 1e-12
x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))
y = 1 / (1.0 + tf.exp(-(tf.matmul(x,W) + b)))
loss1 = -(y_*tf.log(y+eps) + (1-y_) * tf.log( 1 - y + eps))
loss = tf.reduce_mean(loss1)
update = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
data_x = np.array([convert2vec(data[0]), convert2vec(data[1]), convert2vec(data[2]), convert2vec(data[3])])
data_y = np.array([[1],[1],[0],[0]])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,10000):
	sess.run(update, feed_dict = {x:data_x, y_:data_y}) #BGD




#
test1 = "I need you now! Please answer ASAP!"
test2 = "I wanted to hear your thoughts about my plans."
# pdb.set_trace()
print('Prediction for: "' + test1 + '"',
      logistic_fun(np.matmul(np.array([convert2vec(test1)]), sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "' + test2 + '"',
      logistic_fun(np.matmul(np.array([convert2vec(test2)]), sess.run(W)) + sess.run(b))[0][0])