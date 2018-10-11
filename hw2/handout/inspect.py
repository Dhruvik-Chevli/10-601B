import csv
import sys
import os
import math
from itertools import islice

def calculateEntropy(fileDict):
    """
        This function takes the labels returns the label entropy
    """
    entropy = 0
    for key in fileDict:
        if fileDict[key] != 0:
            prob = fileDict[key]/sum(fileDict.values())
            entropy += -1*prob*math.log(prob,2)
    
    return entropy

def calculateError(fileDict):
    """
        This function takes the labels and returns the error rate of the majority vote classifier
    """
    vote = max(fileDict.values())
    errorCount = sum(fileDict.values()) - vote

    return errorCount/sum(fileDict.values())
    
def lineCount(fileInput):
    """
        This function takes in the file and return the number of data points in the dataset
    """
    with open(fileInput, 'r') as infile:
        next(infile)
        for i, l in enumerate(infile):
            pass
    return i+1

def createFileDict(fileInput):
    """
        This function takes in the file and returns the label dictionary
    """
    labels = dict()
    fileLineCount = lineCount(fileInput)

    # construct the label dictionary
    with open(fileInput, 'r') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar = '|')
        for line in islice(filereader, 1, fileLineCount+1):
            label = line[-1]
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1

    return labels

def fileOps(fileInput, fileOutput):
    """
        This function takes the parameter files and returns the output file.
        Assumption: The function assumes that the last element of any data point row will be the label
    """
    labels = createFileDict(fileInput)
    errorRate = calculateError(labels)
    entropy = calculateEntropy(labels)

    # write to output file
    with open(fileOutput, 'w') as outfile:
        outfile.write('entropy: ' + str(entropy) + '\n')
        outfile.write('error: ' + str(errorRate))

def main():
    if(os.stat(str(sys.argv[1])).st_size == 0):
        open(str(sys.argv[2]), "w").close()
    else:
        fileOps(str(sys.argv[1]), str(sys.argv[2]))

if __name__ == "__main__":
    main()