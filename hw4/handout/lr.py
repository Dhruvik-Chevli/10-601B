import sys
import os
import numpy as np
# import time
# import matplotlib.pyplot as plt

def sigmoid(scores):
    # return the sigmoid of the vector
    return 1/(1+np.exp(-scores))

def getLabels(scores):
    # return the label as 0 or 1
    return np.round(sigmoid(scores))[0]

def sparse_dot(X,W, bias):
    # calculate the dot product, given the sparse features
    product = 0.0
    for k,v in X.items():
        product += (v*W[int(k)])
    return product+bias

def sparse_gradient(X, delta, length):
    # calculate the sparse gradient for the log-likelihood
    gradient = np.zeros((length, 1))
    for k,v in X.items():
        gradient[int(k)] = v*delta
    return gradient

def avg_neg_log_likelihood(features, targets, weights, bias):
    final_ll = 0
    for i in range(0, len(features)):
        z = sparse_dot(features[i], weights, bias)
        y = targets[i]
        ll = -y*z + np.log(1+np.exp(z))
        final_ll += ll
    return final_ll/len(features)

def fit(train_features, train_labels, valid_features, valid_labels, weights, bias, learning_rate, num_epochs):
    # train for num_epochs across all training samples
    # train_ll = []
    # valid_ll = []
    for j in range(0, num_epochs):
        for i in range(0, len(train_features)):
            z = sparse_dot(train_features[i], weights, bias)
            delta_update = (sigmoid(z)-train_labels[i])[0]
            weight_gradient = sparse_gradient(train_features[i], delta_update, len(weights))
            bias_gradient = delta_update
            weights = np.subtract(weights, learning_rate*weight_gradient)
            bias = bias - learning_rate*bias_gradient
        # train_ll.append(avg_neg_log_likelihood(train_features, train_labels, weights, bias))
        # valid_ll.append(avg_neg_log_likelihood(valid_features, valid_labels, weights, bias))
    
    # return weights, bias, train_ll, valid_ll
    return weights, bias

def predictError(features, weights, bias, actualLabels):
    # get the predictions of labels for the given feature set and return the error
    # also return the string of labels for output
    labels = [int(getLabels(sparse_dot(x, weights, bias))) for x in features]
    b = [a for a,p in zip(actualLabels, labels) if a!=p]
    accuracy = float(len(b))/len(labels)
    new_labels = map(str, labels)
    string_labels = "\n".join(new_labels)
    string_labels+="\n"
    return labels, string_labels, accuracy

def parseFile(file_in):
    # return labels and feature set from formatted file
    dict_lines = {}
    with open(file_in, 'r') as infile:
        dict_lines = infile.readlines()

    labels = []
    new_dict = []
    for line in dict_lines:
        temp_dict = {}
        line = line.rstrip()
        x = line.split('\t')
        labels.append(int(x[0]))
        for y in x[1:]:
            a = y.split(':')
            temp_dict[a[0]] = int(a[1])
        new_dict.append(temp_dict)
    return labels, new_dict

if __name__ == "__main__":
    """
        Read the arguments and call the function
    """
    # form the zero weight vector
    dict_lines = {}
    with open(str(sys.argv[4]), 'r') as infile:
        dict_lines = infile.readlines()
    weights = np.zeros((len(dict_lines), 1))
    bias = 0

    # parse the formatted files given
    trainlabels, train_dict = parseFile(sys.argv[1])
    validlabels, valid_dict = parseFile(sys.argv[2])
    testlabels, test_dict = parseFile(sys.argv[3])

    # train the function
    # learnedweights, learnedbias, train_ll, valid_ll = fit(train_dict, trainlabels, valid_dict, validlabels, weights, bias, 0.1, int(sys.argv[8]))

    learnedweights, learnedbias = fit(train_dict, trainlabels, valid_dict, validlabels, weights, bias, 0.1, int(sys.argv[8]))

    # x = range(0, int(sys.argv[8]))
    # a = plt.plot(x, train_ll)
    # b = plt.plot(x, valid_ll)
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Negative Log-Likelihood')
    # plt.title('Negative LL vs #Epochs')
    # plt.legend(('Train LL', 'Valid LL'), loc='upper right')
    # plt.show()

    # get the errors and labels
    new_train_labels, train_string, trainError = predictError(train_dict, learnedweights, learnedbias, trainlabels)
    new_test_labels, test_string, testError = predictError(test_dict, learnedweights, learnedbias, testlabels)
    
    # write outputs
    with open(sys.argv[5], 'w') as outfile:
        outfile.write(train_string)

    with open(sys.argv[6], 'w') as outfile:
        outfile.write(test_string)

    with open(sys.argv[7], 'w') as outfile:
        outfile.write('error(train): ' + str(trainError) + '\n')
        outfile.write('error(test): ' + str(testError))