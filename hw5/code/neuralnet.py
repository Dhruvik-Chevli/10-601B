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
            z = list(map(float, row[1:]))
            features[i,1:] = z
            i += 1

    return labels, features

def create_outputs(filename, mean_train_ce, mean_test_ce, train_err, test_err, train_out, test_out):
    pass

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
            self.alpha = np.zeros((hidden_units,self.train_x.shape[1]))
            self.alpha[:,1:] = np.random.uniform(-0.1,0.1, (hidden_units, self.train_x.shape[1]-1))
            self.beta = np.zeros((10, hidden_units+1))
            self.beta[:,1:] = np.random.uniform(-0.1,0.1, (10, hidden_units))
        elif init_strategy == 2:
            self.alpha = np.zeros((hidden_units,self.train_x.shape[1]))
            self.beta = np.zeros((10,hidden_units+1))
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.train_error = 0.0
        self.test_error = 0.0
        self.train_cross_entropy = []
        self.test_cross_entropy = []
        self.mean_train_cross_entropy = np.zeros((num_epochs,1))
        self.mean_test_cross_entropy = np.zeros((num_epochs,1))

    def sigmoid_forward(self, a):
        return 1/(1+np.exp(-a))

    def sigmoid_backward(self, b, grad_b):
        return np.multiply(grad_b.ravel(), np.multiply(b,1-b))

    def softmax_forward(self, a):
        return np.divide(np.exp(a),np.sum(np.exp(a)))

    def softmax_backward(self,b,grad_b):
        grad_b = np.reshape(grad_b, (grad_b.shape[0],1))
        b_dash = np.reshape(b, (b.shape[0],1))
        return np.matmul(grad_b.T, np.subtract(np.diag(b), np.matmul(b_dash,b_dash.T)))

    def linear_forward(self, a, weights):
        return np.matmul(weights,a)

    def linear_backward(self, a, weights, grad_b):
        a = np.reshape(a, (a.shape[0],1))
        return np.matmul(grad_b.T, a.T), np.matmul(weights.T, grad_b.T)

    def cross_entropy_forward(self, a, a_hat):
        return -np.matmul(a.T,np.log(a_hat))

    def cross_entropy_backward(self, a, a_hat):
        return -np.divide(a,a_hat)

    def feed_forward(self, x, y):
        a = self.linear_forward(x, self.alpha)
        z = self.sigmoid_forward(a)
        z = np.append(1,z)
        b = self.linear_forward(z, self.beta)
        y_hat = self.softmax_forward(b)
        J = self.cross_entropy_forward(y,y_hat)
        return x,a,z,b,y_hat,J

    def back_propagate(self, x, y, args):
        x,a,z,b,y_hat,J = args
        gy_hat = self.cross_entropy_backward(y,y_hat)
        gb = self.softmax_backward(y_hat,gy_hat)
        gbeta, gz = self.linear_backward(z,self.beta,gb)
        ga = self.sigmoid_backward(z,gz)
        ga = ga[1:]
        ga = np.reshape(ga, (1,ga.shape[0]))
        galpha, gx = self.linear_backward(x,self.alpha,ga)
        return galpha, gbeta

    def fit(self):
        for i in range(0, self.num_epochs):
            for j in range(0, self.train_x.shape[0]):
                x,y = self.train_x[j,:], self.train_y[j,:]
                x,a,z,b,y_hat,J = self.feed_forward(x,y)
                galpha, gbeta = self.back_propagate(x,y,[x,a,z,b,y_hat,J])
                self.alpha = self.alpha - self.learning_rate*galpha
                self.beta = self.beta - self.learning_rate*gbeta

            self.train_cross_entropy = []
            for k in range(0, self.train_x.shape[0]):
                x,y = self.train_x[k,:], self.train_y[k,:]
                x,a,z,b,y_hat,J = self.feed_forward(x,y)
                self.train_cross_entropy.append(J)

            self.test_cross_entropy = []
            for m in range(0, self.test_x.shape[0]):
                x,y = self.test_x[m,:], self.test_y[m,:]
                x,a,z,b,y_hat,J = self.feed_forward(x,y)
                self.test_cross_entropy.append(J)

            self.mean_train_cross_entropy[i] = sum(self.train_cross_entropy)/len(self.train_cross_entropy)
            self.mean_test_cross_entropy[i] = sum(self.test_cross_entropy)/len(self.test_cross_entropy)
    
    def actual_predict(self, x,y):
        predict_y = np.zeros((x.shape[0], 1))
        # print (x[0,:], y[0,:])
        for i in range(0,y.shape[0]):
            x,a,z,b,y_hat,J = self.feed_forward(x[i,:], y[i,:])
            predict_y[i] = np.argmax(y_hat)
        return predict_y

    def predict(self, type_data='test'):
        if type_data == 'train':
            self.predict_train = np.zeros((self.train_x.shape[0],1))
            for i in range(0, self.train_x.shape[0]):
                x,y = self.train_x[i,:], self.train_y[i,:]
                x,a,z,b,y_hat,J = self.feed_forward(x, y)
                self.predict_train[i] = np.argmax(y_hat)
        else:
            self.predict_test = np.zeros((self.test_x.shape[0],1))
            for i in range(0, self.test_x.shape[0]):
                x,y = self.test_x[i,:], self.test_y[i,:]
                x,a,z,b,y_hat,J = self.feed_forward(x, y)
                self.predict_test[i] = np.argmax(y_hat)
    
    def get_actual_error(self, actual_labels, predicted_labels):
        count = 0
        for a,p in zip(actual_labels, predicted_labels):
            actual = float(np.argwhere(a==1)[0])
            predicted = float(p[0])
            if actual == predicted:
                count+=1
        
        return 1-count/len(predicted_labels)

    def get_error(self, type_data = 'test'):
        if type_data == 'train':
            self.train_error = self.get_actual_error(self.train_y, self.predict_train)
        else:
            self.test_error = self.get_actual_error(self.test_y, self.predict_test)

    def create_outputs(self, train_out, test_out, metrics_out):
        train_string = ""
        for a in self.predict_train:
            x = int(a[0])
            train_string = train_string + str(x) + "\n"

        train_string = train_string.rstrip()

        test_string = ""
        for a in self.predict_test:
            x = int(a[0])
            test_string = test_string + str(x) + "\n"

        test_string = test_string.rstrip()

        with open(train_out, 'w') as outfile:
            outfile.writelines(train_string)

        with open(test_out, 'w') as outfile:
            outfile.writelines(test_string)

        metrics_string = ""
        for i in range(0, self.num_epochs):
            metrics_string = metrics_string + "epoch=" + str(i+1) + " crossentropy(train): " + str(self.mean_train_cross_entropy[i][0]) + "\n"
            metrics_string = metrics_string + "epoch=" + str(i+1) + " crossentropy(test): " + str(self.mean_test_cross_entropy[i][0]) + "\n"

        # metrics_string = metrics_string.rstrip()
        metrics_string = metrics_string + "error(train): " + str(self.train_error) + "\n"
        metrics_string = metrics_string + "error(test): " + str(self.test_error)
        # print (metrics_string)

        with open(metrics_out, 'w') as outfile:
            outfile.writelines(metrics_string)

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

    model = NeuralNetwork(train_features, train_labels, test_features, test_labels, hidden_units, init_flag, num_epochs, learning_rate)
    model.fit()
    model.predict('train')
    model.predict('test')
    model.get_error('train')
    model.get_error('test')

    model.create_outputs(train_out, test_out, metrics_out)