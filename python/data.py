import csv
import numpy as np
import re

def load_data(path):
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
    params = {}
    p = re.compile(" *([^ :])+[ :]+(.+) *#*")
    for line in string.splitlines():
        m = p.match(line)
        if m is None:
            continue
        key = m.group(1)
        val = m.group(2)
        params[key] = val
    return params

if __name__ == '__main__':
    s = "a: 150\nb:  2.31-2.32\nc   \opt\Ime Projekta"
    print(parse_params(s))
