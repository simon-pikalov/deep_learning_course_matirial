import numpy as np

if __name__ == '__main__':
    galaxy_data = np.array([[2,70],[3,110],[4,165],[6,390],[7,550]])
    w = 0
    b = 0
    alpha = 0.01
    for iteration in range(10000):
        gradient_b = np.mean(1*((w*galaxy_data[:,0]+b) - galaxy_data[:,1]))
        gradient_w = np.mean(galaxy_data[:,0]*((w*galaxy_data[:,0]+b)-galaxy_data[:,1]))
        b -= alpha*gradient_b
        w -= alpha*gradient_w
        if iteration % 200 == 0 :
            print("it:%d,  grad_w:%.3f, grad_b:%.3f, w:%.3f, b:%.3f" %(iteration, gradient_w, gradient_b, w, b))
    print("Estimated price for Galaxy S5: ", w*5 + b)