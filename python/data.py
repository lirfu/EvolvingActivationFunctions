import csv
import numpy as np


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

def parse_params(string):
    return {} #TODO implement params parsing