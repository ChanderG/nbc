#! /usr/bin/python

import math
import numpy as np
import random

def loadData(filename):
    """Load data from filename and return in dict format."""
    rawdata = open(filename).read()
    rawdata = filter(lambda x: x != '', rawdata.split("\n"))

    data = []
    for line in rawdata:
        d = {}
        items = filter(lambda x: x != '', line.split(" "))
        d['class'] = items[0]
        for i in items[1:]:
            kv = i.split(":")
            d[kv[0]] = kv[1]
        data.append(d)
    return data

def calculateProbability(x, mean, stdev):
    """ Gaussian distribution. """
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def createNBClassifier(data):
    """ Create a Naive Bayes classifier. 

    Returns a dict with features and probability of T/F for each value of each feature.
    """

    # for each feature, need to calculate probability of True/False

    # get the 2 classes
    classes = set([])
    for d in data:
        classes.add(d['class'])
        if len(classes) == 2:
            break

    # simple set labels
    true_label = classes.pop()
    false_label = classes.pop()

    # for each feature we need to calculate probabilities of true/false
    keys = filter( lambda x: x != 'class', data[0].keys())

    classifier = {}
    totalnos = len(data)

    # does a loop over all elements in list for every key
    # can be optimized to one loop, TODO

    for k in keys:
        probset = {}
        probset['true'] = {}
        probset['false'] = {}

        for d in data:
            if d['class'] == true_label:
                probset['true'][d[k]] = probset['true'].get(d[k], 0) + 1
                probset['false'][d[k]] = probset['false'].get(d[k], 0) + 0
            else:
                probset['false'][d[k]] = probset['false'].get(d[k], 0) + 1
                probset['true'][d[k]] = probset['true'].get(d[k], 0) + 0

        # arbitrary cutoff to decide when the number of keys are too many
        if len(probset['true'].keys() + probset['false'].keys()) > 0.3*len(data):
            # too many keys present
            # discrete probability does not make sense
            # we need to model a gaussian distribution
            #probset = {}
            probset['gaussian'] = True

            # obtain mean and standard deviation
            true_nos = []
            false_nos = []
            for d in data:
                if d['class'] == true_label:
                    true_nos.append(float(d[k]))
                else:
                    false_nos.append(float(d[k]))
            
            true_nos = np.array(true_nos)
            false_nos = np.array(false_nos)

            probset['true_mean'] = float(np.mean(true_nos))
            probset['true_std'] = float(np.std(true_nos))

            probset['false_mean'] = float(np.mean(false_nos))
            probset['false_std'] = float(np.std(false_nos))

        else: 
            # use ordinary distribution
            probset['gaussian'] = False

            # convert to probabilities
            for p in probset['true'].keys():
                probset[p] = float(probset['true'][p])/totalnos
            for p in probset['false'].keys():
                probset[p] = float(probset['false'][p])/totalnos

        # add it master dict
        classifier[k] = probset


    # add true and false labels
    classifier['true'] = true_label
    classifier['false'] = false_label

    #print classifier
    return classifier

def runNBClassifier(classifier, testdata):
    """ Run the classifier and return the class."""

    prob_true = 1.0
    prob_false = 1.0

    for k in testdata.keys():
       if k == 'class':
           continue
       if classifier[k]['gaussian'] == False:
           prob_true = prob_true * classifier[k]['true'][testdata[k]] 
           prob_false = prob_false * classifier[k]['false'][testdata[k]]

       else:
           # alternative method to calculate probability
           prob_true = calculateProbability(float(testdata[k]), classifier[k]['true_mean'], classifier[k]['true_std'])
           prob_false = calculateProbability(float(testdata[k]), classifier[k]['false_mean'], classifier[k]['false_std'])


    if prob_true > prob_false:
        return classifier['true']
    else:
        return classifier['false']

def testNBClassifier(classifier, testdata):
    """ Give correctly classified percentage.

    testdata -- is a list of data
    """
    correct = 0
    total = len(testdata)

    for t in testdata:
        if t['class'] == runNBClassifier(classifier, t):
            correct  = correct + 1

    return correct*100.0/total

def kFoldCrossValidation(data, k):
    """ Return k-fold cross validation accuracy. 
    
    data -- list of data points
    k - fold number
    """
    random.shuffle(data)
    res = []
    _k = len(data) // k

    for i in range(0, len(data), _k):
        # split the data into test and training
        test = data[i:min(i+_k, len(data))]
        training = data[:i] + data[min(i+_k, len(data)):]

        classifier = createNBClassifier(training)
        res.append(testNBClassifier(classifier, test))

    res = np.array(res)
    return np.mean(res)

def main():
    """Main entry point."""
    data = loadData("breast-cancer.libsvm")
    classifier = createNBClassifier(data)

    # sample test
    """
    for i in range(0, 100):
        print i
        print "Actual : ", data[i]['class']
        print "Predicted :", runNBClassifier(classifier, data[i])
    """

    print "k fold cross validation accuracy: "

    print "3: ", kFoldCrossValidation(data, 3)
    print "5: ", kFoldCrossValidation(data, 5)
    print "10:", kFoldCrossValidation(data, 10)

if __name__ == "__main__":
    main()
