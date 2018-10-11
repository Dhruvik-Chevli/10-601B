import csv
import numpy as np
from decisionTree import *

def fileOps(trainInput, testInput):
    colNames, data = handleData(trainInput)
    depthRange = len(data[0])-1

    trainErrorVector = []
    testErrorVector = []
    depthVector = []

    vote, trainError = maxDepthZeroOps(trainInput, None, None)
    majVote, testError = maxDepthZeroOps(testInput, None, vote)
    depthVector.append(0)
    trainErrorVector.append(trainError)
    testErrorVector.append(testError)

    for i in range(1, depthRange+1):
        tree = trainDataOps(trainInput, i)
        trainError = dataOps(trainInput, tree, None)
        testError = dataOps(testInput, tree, None)
        depthVector.append(i)
        trainErrorVector.append(trainError)
        testErrorVector.append(testError)

    print (depthVector, trainErrorVector, testErrorVector)

if __name__ == "__main__":
    fileOps(str(sys.argv[1]), str(sys.argv[2]))