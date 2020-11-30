import numpy as np
data_x = np.array([[1,28,4],[1,60,34],[1,25,3],[0,54,20],[0,24,2],[0,39,12],[0,30,4],[1,36,10],[1,26,1],[0,44,9]])
data_y = np.array([1,1,1,1,1,1,1,0,0,0])
#The Data
# Female, 28 years old, 4 years of experience
# Female, 60 years old, 34 years of experience
# Female, 25 years old, 3 year of experience
# Male, 54 years old, 20 years of experience
# Male, 24 years old, 2 years of experience
# Male, 39 years old, 12 years of experience
# Male, 30 years old, 4 years of experience
# Unemployed users:
# Female, 36 years old 10 years of experience
# Female, 26 years old 1 year of experience
# Male, 44 years old, 9 years of experience

def h(x,w,b):
    return 1 / (1+np.exp(-(np.dot(x,w) + b)))

w = np.array([0.,0,0])
b = 0
alpha = 0.001
for iteration in range(100000):
    gradient_b = np.mean(1*((h(data_x,w,b))-data_y))
    gradient_w = np.dot((h(data_x,w,b)-data_y), data_x)*1/len(data_y)
    b -= alpha*gradient_b
    w -= alpha*gradient_w

print(w,b)
print("User [1, 49, 8] prob of working: ", h(np.array([[1, 49, 8]]),w,b))
print("User [0, 29, 3] prob of working: ", h(np.array([[0, 29, 3]]),w,b))
print("User [1, 29, 3] prob of working: ", h(np.array([[1, 29, 3]]),w,b))