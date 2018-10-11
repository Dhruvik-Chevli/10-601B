import os
import sys
import math
import numpy as np
import csv
from math import log
import operator
from inspect import calculateEntropy, calculateError 

class DecisionTree:
    """
        This is the class that implements a decision tree
    """
    def __init__(self, col = -1, value = None, results = None, vote = None, depth = None, leftBranch = None, rightBranch = None):
        self.col = col
        self.value = value
        self.results = results
        self.vote = vote
        self.depth = depth
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch

def getLabels(data):
    """
        This function takes in the dataset and returns the dictionary containing labels
    """
    results = dict()
    for row in data:
        label = row[-1]
        if label not in results:
            results[label] = 1
        else:
            results[label] += 1
    
    return results

def splitDataset(data, col, valuetoDivide):
    """
        This function takes in the data and the value to divide on and splits the dataset into two divisions (postiive and negative),
        which will be assigned to the left/right childs of the dataset equivalently
    """
    positiveSet = [row for row in data if row[col] == valuetoDivide]
    negativeSet = [row for row in data if not row[col] == valuetoDivide]

    return (positiveSet, negativeSet)

def calcMI(data, set1, set2):
    """
        This function takes in the data and returns the mutual information with respect to the split attribute
    """
    # get entropy of Y
    labels = getLabels(data)
    labelEntropy = calculateEntropy(labels)

    # get conditional entropy of Y | attr
    labels1 = getLabels(set1)
    labels2 = getLabels(set2)
    prob = float(len(set1)) / len(data)
    conditionalEntropy = prob*calculateEntropy(labels1) + (1-prob)*calculateEntropy(labels2)
    
    # return entr(Y) - entr(Y|attr)
    return labelEntropy - conditionalEntropy

def buildDecisionTree(data, colNames, depth, maxDepth):
    """
        This function takes in a dataset and returns the resultant decision tree that has been built
    """
    if len(data) == 0:
        return DecisionTree()
    
    highestMI = 0
    bestAttribute = None
    bestSplit = None

    numColumns = len(data[0])-1
    for col in range(0, numColumns):
        # check for every column if the mutual information is the highest
        columnValue = [row[col] for row in data]
        columnValue = list(set(columnValue))

        for value in columnValue:
            (set1, set2) = splitDataset(data, col, value)
            # compute the mutual information
            gain = calcMI(data, set1, set2)

            # assign the best attribute and the best split according to mutual information
            if gain > highestMI and len(set1) > 0 and len(set2) > 0:
                highestMI = gain
                bestAttribute = (col, value)
                bestSplit = (set1, set2)

    # check if the depth condition is satisfied, if not, truncate tree with corresponding leaf nodes
    # also check if the mutual information is greater than zero
    if highestMI > 0 and depth < maxDepth:
        depth = depth+1
        leftBranch = buildDecisionTree(bestSplit[0], colNames, depth, maxDepth)
        rightBranch = buildDecisionTree(bestSplit[1], colNames, depth, maxDepth)
        return DecisionTree(col = bestAttribute[0], value = bestAttribute[1], depth = depth, leftBranch=leftBranch, rightBranch=rightBranch)
    else:
        # store vote and split of data at leaf node only
        labels = getLabels(data)
        vote = max(labels.items(), key=operator.itemgetter(1))[0]
        return DecisionTree(results = labels, vote = vote, depth = depth)

def prettyPrint(data, colNames, decisionTree):
    """
        Pretty print the obtained decision tree
    """
    def toString(decisionTree, indent = '| '):
        if decisionTree.results != None:
            """
                Leaf nodes ideally exist at depth + 1
            """
            return '| ' + str(decisionTree.results)
        else:
            decision = '| %s : %s' % (colNames[decisionTree.col], decisionTree.value)
            leftBranch = indent + toString(decisionTree.leftBranch, indent + '| ')
            rightBranch = indent + toString(decisionTree.rightBranch, indent + '| ')
            return (decision + '\n' + leftBranch + '\n' + rightBranch)

    print(str(getLabels(data)) + "\n" + toString(decisionTree))

def classify(datapoint, tree):
    """
        This function takes in any data and returns the classification of the given sample against the tree
    """
    if tree.results != None:
        return tree.vote
    else:
        v = datapoint[tree.col]
        branch = None
        if v == tree.value:
            branch = tree.leftBranch
        else:
            branch = tree.rightBranch

        return classify(datapoint, branch)

def getClassificationLabelsAndErrorRate(data, tree):
    """
        This function takes in the data and returns the error rate for classification
    """
    errorRate = 0
    numSamples = len(data)
    classificationVote = []
    for datapoint in data:
        label = datapoint[-1]
        observation = datapoint[:-1]
        vote = classify(observation, tree)
        classificationVote.append(vote)
        if vote != label:
            errorRate += 1

    errorRate = errorRate/numSamples
    return (classificationVote, errorRate)

def handleData(dataInput):
    with open(dataInput, 'r') as infile:
        data = list(list(rec) for rec in csv.reader(infile, delimiter=','))

    columnNames = data[0]    
    data = data[1:]
    return (columnNames, data)

def trainDataOps(dataInput, maxDepth):
    """
        This function handles the forming of the decision tree and returns the actual tree
    """

    (columnNames, data) = handleData(dataInput)
    decisionTree = buildDecisionTree(data, columnNames, 0, maxDepth)
    prettyPrint(data, columnNames, decisionTree)
    return decisionTree

def dataOps(dataInput, tree, dataOutput = None):
    """
        This function takes in the data and returns the error for the given data
    """
    (columnNames, data) = handleData(dataInput)
    labels, error = getClassificationLabelsAndErrorRate(data, tree)

    # write labels to .labels file
    if dataOutput:
        with open(dataOutput, 'w') as outfile:
            outputString = ""
            for key in labels:
                outputString += key + "\n"

            outfile.write(outputString.rstrip())

    return error

def maxDepthZeroOps(dataInput, dataOutput = None, majClassVote = None):
    """
        This function handles the special case that max depth is zero (majority vote classifier)
    """
    (columnNames, data) = handleData(dataInput)
    labels = getLabels(data)
    if majClassVote:
        error = 0
        for datapoint in data:
            if datapoint[-1] != majClassVote:
                error += 1
        error = error/len(data)
    else:
        error = calculateError(labels)

    vote = max(labels.items(), key=operator.itemgetter(1))[0]
    classificationVote = []
    for datapoint in data:
        if majClassVote:
            classificationVote.append(majClassVote)
        else:
            classificationVote.append(vote)

    if dataOutput:
        with open(dataOutput, 'w') as outfile:
            outputString = ""
            for key in classificationVote:
                outputString += key + "\n"

            outfile.write(outputString.rstrip())

    if majClassVote:
        return majClassVote, error
    else:
        return vote, error

def fileOps(trainInput, testInput, maxDepth, trainOutput, testOutput, metricsFile):
    """
        This function takes the parameter files and returns the output file.
    """
    
    if maxDepth < 0:
        return
    if maxDepth == 0:
        vote, trainError = maxDepthZeroOps(trainInput, trainOutput, None)
        majVote, testError = maxDepthZeroOps(testInput, testOutput, vote)
    else:
        tree = trainDataOps(trainInput, maxDepth)
        trainError = dataOps(trainInput, tree, trainOutput)
        print ("testInput: " + testInput)
        testError = dataOps(testInput, tree, testOutput)

    # write error rates to metrics file
    with open(metricsFile, 'w') as outfile:
        outfile.write('error(train): ' + str(trainError) + '\n')
        outfile.write('error(test): ' + str(testError))
    
if __name__ == "__main__":
    if(os.stat(str(sys.argv[1])).st_size == 0):
        open(str(sys.argv[4]), "w").close()
        open(str(sys.argv[5]), "w").close()
        open(str(sys.argv[6]), "w").close()
    else:
        fileOps(str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]))