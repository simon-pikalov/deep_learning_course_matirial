from datetime import datetime
from datetime import timedelta

import numpy as np


def simple_quadraric():
    start_time = (datetime.now())
    data_x = np.array([[2, 4], [3, 9], [4, 16], [6, 36], [7, 49]])
    data_y = np.array([70, 110, 165, 390, 550])
    w1 = 0
    w2 = 0
    b = 0
    alpha = 0.001
    for iteration in range(1000000):
        gradient_b = np.mean(1 * (data_y - (w1 * data_x[:, 0] + w2 * data_x[:, 1] + b)))
        gradient_w1 = np.dot((data_y - (w1 * data_x[:, 0] + w2 * data_x[:, 1] + b)), data_x[:, 0]) * 1.0 / len(data_y)
        gradient_w2 = np.dot((data_y - (w1 * data_x[:, 0] + w2 * data_x[:, 1] + b)), data_x[:, 1]) * 1.0 / len(data_y)
        b += alpha * gradient_b
        w1 += alpha * gradient_w1
        w2 += alpha * gradient_w2

    print("Estimated price for Galaxy S5: ", np.dot(np.array([5, 25]), np.array([w1, w2])) + b)
    print("Estimated price for Galaxy S1: ", np.dot(np.array([1, 1]), np.array([w1, w2])) + b)
    #end_time = str(datetime.now())
    #print("time simple_quadraric took :%.lf  "%(end_time - timedelta(start_time)))

def weights_as_vectors():
    start_time = (datetime.now())
    data_x = np.array([[2, 4], [3, 9], [4, 16], [6, 36], [7, 49]])
    data_y = np.array([70, 110, 165, 390, 550])

    w = np.array([0., 0])
    b = 0
    alpha = 0.001
    for iteration in range(1000000):
        gradient_b = np.mean(1 * (data_y - (np.dot(data_x, w) + b)))
        gradient_w = 1.0 / len(data_y) * np.dot((data_y - (np.dot(data_x, w) + b)), data_x)
        b += alpha * gradient_b
        w += alpha * gradient_w

    print("Estimated price for Galaxy S5: ", np.dot(np.array([5, 25]), w) + b)
    print("Estimated price for Galaxy S1: ", np.dot(np.array([1, 1]), w) + b)
    #end_time = str(datetime.now())
    #print("time simple_quadraric took :"  ,str(end_time - timedelta(start_time)))


# start 08:10:45.043496
# Estimated price for Galaxy S5:  263.63638156011774
# Estimated price for Galaxy S1:  71.81812246777238
# end  08:11:11.364839

# start  08:11:11.364857
# Estimated price for Galaxy S5:  263.63638156011774
# Estimated price for Galaxy S1:  71.81812246777238
# end  08:11:28.951841
if __name__ == '__main__':
    simple_quadraric()
    weights_as_vectors()

