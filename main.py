#! /usr/bin/python

def loadData(filename):
    """Load data from filename and return i dict format."""
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

def main():
    """Main entry point."""
    data = loadData("breast-cancer.libsvm")

if __name__ == "__main__":
    main()
