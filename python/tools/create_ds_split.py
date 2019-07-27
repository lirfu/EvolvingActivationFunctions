#!/usr/bin python

import re
import sys
import os
from os import path
import collections
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

feat_file = sys.argv[1]
lab_file = sys.argv[2]
perc = float(sys.argv[3])

# Load matrices.
features = np.loadtxt(feat_file, delimiter=' ')
labels = np.loadtxt(lab_file, delimiter=' ')

# Split dataset.
x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=perc, random_state=42)

# Draw distributions.
plt.figure()
plt.subplot(2,1,1)
labels, values = zip(*Counter(y_train).items())
indexes = np.arange(len(labels))
plt.bar(labels, values, 1)
plt.title('train')

plt.subplot(2,1,2)
labels, values = zip(*Counter(y_test).items())
indexes = np.arange(len(labels))
plt.bar(labels, values, 1)
plt.title('test')

plt.show()

# Store split datasets.
x_train_name = re.sub('all', 'train', str(feat_file))
x_test_name = re.sub('all', 'test', str(feat_file))

y_train_name = re.sub('all', 'train', str(lab_file))
y_test_name = re.sub('all', 'test', str(lab_file))

np.savetxt(x_train_name, x_train, delimiter=',')
np.savetxt(x_test_name, x_test, delimiter=',')
np.savetxt(y_train_name, y_train, delimiter=',')
np.savetxt(y_test_name, y_test, delimiter=',')

