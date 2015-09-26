#! /usr/bin/python

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

        for d in data:
            if d['class'] == true_label:
                probset[d[k]] = probset.get(d[k], 0) + 1
            else:
                probset[d[k]] = probset.get(d[k], 0) + 0

        #print "Feature ", k, " has ", len(probset.keys())
        #if len(probset.keys()) < 

        # convert to probabilities
        for p in probset.keys():
            probset[p] = float(probset[p])/totalnos

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
       prob_true = prob_true * classifier[k][testdata[k]] 
       prob_false = prob_false * (1.0 - classifier[k][testdata[k]])

    if prob_true > prob_false:
        return classifier['true']
    else:
        return classifier['false']

def main():
    """Main entry point."""
    data = loadData("breast-cancer.libsvm")
    classifier = createNBClassifier(data)

    for i in range(0, 100):
        print i
        print "Actual : ", data[i]['class']
        print "Predicted :", runNBClassifier(classifier, data[i])

if __name__ == "__main__":
    main()
