#!/usr/bin python

import numpy as np
import sklearn
import csv
import sys
from sklearn.metrics import classification_report

def load(path):
    with open(path, 'r') as f:
        data = []
        reader = csv.reader(f, delimiter=',')
        t = False
        for l in reader:
            if len(l) > 0 and l[0] == "@data":
                t = True
                continue
            if t:
                d = []
                for v in l:
                    d.append(float(v))
                data.append(d)
        data = np.array(data)
        return data[:, 0:-1], data[:, -1]

x, y = load(sys.argv[1])
with open(sys.argv[2], 'rb') as f:
    pred = np.load(f)

print('True shape:', y.shape)
print('Prediction shape:', pred.shape)

print(classification_report(y, pred, np.arange(256), digits=3))
