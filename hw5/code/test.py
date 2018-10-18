import numpy as np
import math

# alpha = np.array([[1,1,2,-3,0,1,-3],[1,3,1,2,1,0,2],[1,2,2,2,2,2,1],[1,1,0,2,1,-2,2]])
alpha = np.array([[1,10,20,-30,0,10,-30],[1,30,10,20,10,0,20],[1,20,20,20,20,20,10],[1,10,0,20,10,-20,20]])
# print (np.linalg.norm(alpha))
beta = np.array([[1,1,2,-2,1],[1,1,-1,1,2],[1,3,1,-1,1]])
# print (np.linalg.norm(beta))
gamma = 0.01 

def sigmoid_forward(a):
    return 1/(1+np.exp(-a))

def sigmoid_backward(b, grad_b):
    return np.multiply(grad_b.ravel(), np.multiply(b,1-b))

def softmax_forward(a):
    return np.divide(np.exp(a),np.sum(np.exp(a)))

def softmax_backward(b,grad_b):
    grad_b = np.reshape(grad_b, (grad_b.shape[0],1))
    b_dash = np.reshape(b, (b.shape[0],1))
    return np.matmul(grad_b.T, np.subtract(np.diag(b), np.matmul(b_dash,b_dash.T)))

def linear_forward(a, weights):
    return np.matmul(weights,a)

def linear_backward(a, weights, grad_b):
    a = np.reshape(a, (a.shape[0],1))
    return np.matmul(grad_b.T, a.T), np.matmul(weights.T, grad_b.T)

def cross_entropy_forward(a, a_hat):
    return -np.matmul(a.T,np.log(a_hat))

def cross_entropy_forward_reg(a, a_hat):
    return -np.matmul(a.T,np.log(a_hat)) + gamma*(np.linalg.norm(alpha)**2) + gamma*(np.linalg.norm(beta)**2)

def cross_entropy_backward(a, a_hat):
    return -np.divide(a,a_hat)

def feed_forward(x, y):
    a = linear_forward(x, alpha)
    z = sigmoid_forward(a)
    z = np.append(1,z)
    b = linear_forward(z, beta)
    y_hat = softmax_forward(b)
    # J = cross_entropy_forward(y,y_hat)
    J = cross_entropy_forward_reg(y,y_hat)
    return x,a,z,b,y_hat,J

def back_propagate(x, y, args):
    x,a,z,b,y_hat,J = args
    gy_hat = cross_entropy_backward(y,y_hat)
    gb = softmax_backward(y_hat,gy_hat)
    gbeta, gz = linear_backward(z,beta,gb)
    ga = sigmoid_backward(z,gz)
    ga = ga[1:]
    ga = np.reshape(ga, (1,ga.shape[0]))
    galpha, gx = linear_backward(x,alpha,ga)
    return galpha, gbeta

x = np.array([1,1,1,0,0,1,1])
y = np.array([0,1,0])
# for i in range(0,100):
x,a,z,b,y_hat,J = feed_forward(x,y)
print (J)
galpha, gbeta = back_propagate(x,y,[x,a,z,b,y_hat,J])
alpha = alpha - galpha
beta = beta - gbeta
print (alpha)
print (beta)
# print (beta)