import numpy as np
import math
import csv
import sys

def process_data(filename):
    dict_lines = []
    with open(filename) as infile:
        dict_lines = infile.readlines()

    length = len(dict_lines)
    labels = np.zeros((length, 10))
    features = np.ones((length, 129))

    with open(filename) as infile:
        csv_reader = csv.reader(infile, delimiter = ',', quotechar='|')
        i = 0
        for row in csv_reader:
            labels[i,int(row[0])] = 1
            features[i,1:] = list(map(float,row[1:]))
            i += 1

    return labels, features

class NeuralNetwork:
    def __init__ (self, train_x, train_y, test_x, test_y, hidden_units, init_strategy, num_epochs, learning_rate):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.predict_train = []
        self.predict_test = []
        assert(train_x.shape[1] == test_x.shape[1])
        if init_strategy == 1:
            self.alpha = np.random.rand(self.train_x.shape[1], hidden_units+1)
            self.beta = np.random.rand(hidden_units+1, 10)
        elif init_strategy == 2:
            self.alpha = np.zeros((self.train_x.shape[1], hidden_units+1))
            self.beta = np.zeros((hidden_units+1, 10))
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.train_error = 0.0
        self.test_error = 0.0

    def sigmoid_forward(self, a):
        return 1/(1+np.exp(-a))

    def sigmoid_backward(self, a, b, grad_b):
        return np.multiply(grad_b, np.matmul(b,np.subtract(1,b)))

    def softmax_forward(self, a):
        return np.exp(a)/np.sum(np.exp(a))

    def softmax_backward(self, a,b,grad_b):
        return np.matmul(grad_b.T, np.subtract(np.diag(b), np.matmul(b,b.T)))

    def linear_forward(self, a, weights):
        return np.matmul(weights.T,a)

    def linear_backward(self, a, weights, b, grad_b):
        grad_b = np.reshape(grad_b, (grad_b.shape[0],1))
        a = np.reshape(a, (a.shape[0],1))
        return np.matmul(grad_b, a.T).T, np.matmul(weights, grad_b)

    def cross_entropy_forward(self, a, a_hat):
        return -np.dot(a,np.log(a_hat))

    def cross_entropy_backward(self, a, a_hat, b, grad_b):
        return -np.multiply(grad_b, np.divide(a,a_hat))

    def feed_forward(self, x, y):
        a = self.linear_forward(x, self.alpha)
        z = self.sigmoid_forward(a)
        b = self.linear_forward(z, self.beta)
        y_hat = self.softmax_forward(b)
        J = self.cross_entropy_forward(y,y_hat)
        return x,a,z,b,y_hat,J

    def back_propagate(self, x, y, args):
        x,a,z,b,y_hat,J = args
        gj = 1
        gy_hat = self.cross_entropy_backward(y,y_hat,J,gj)
        gb = self.softmax_backward(b,y_hat,gy_hat)
        gbeta, gz = self.linear_backward(z,self.beta,b,gb)
        ga = self.sigmoid_backward(a,z,gz)
        galpha, gx = self.linear_backward(x,self.alpha,a,ga)
        return galpha, gbeta

    def fit(self):
        for i in range(self.num_epochs):
            for j in range(0, self.train_x.shape[0]):
                x,y = self.train_x[j,:], self.train_y[j,:]
                x,a,z,b,y_hat,J = self.feed_forward(x,y,)
                galpha, gbeta = self.back_propagate(x,y,[x,a,z,b,y_hat,J])
                self.alpha = self.alpha - self.learning_rate*galpha
                self.beta = self.beta - self.learning_rate*gbeta
    
    def actual_predict(self, x,y):
        predict_y = np.zeros((x.shape[0], 1))
        for i in range(y.shape[0]):
            print (x[i])
            print (y[i])
            x,a,z,b,y_hat,J = self.feed_forward(x[i,:], y[i,:])
            predict_y[i] = np.argmax(y_hat)
        return predict_y

    def predict(self, type_data='test'):
        if type_data == 'train':
            self.predict_train = self.actual_predict(self.train_x,self.train_y)
        else:
            self.predict_test = self.actual_predict(self.test_x,self.test_y)
    
    def get_actual_error(self, actual_labels, predicted_labels):
        b = [a for a,p in zip(actual_labels, predicted_labels) if a!=p]
        return float(len(b))/len(actual_labels)

    def get_error(self, type_data = 'test'):
        if type_data == 'train':
            self.train_error = self.get_actual_error(self.train_y, self.predict_train)
        else:
            self.test_error = self.get_actual_error(self.test_y, self.predict_test)
    

if __name__ == "__main__":
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epochs = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    train_labels, train_features = process_data(train_data)
    test_labels, test_features = process_data(test_data)

    model = NeuralNetwork(train_features, train_labels, test_features, test_labels,hidden_units, init_flag, num_epochs, learning_rate)
    model.fit()
    model.predict('train')
    model.predict('test')
    model.get_error('train')
    model.get_error('test')

    print (model.train_error)
    print (model.test_error)