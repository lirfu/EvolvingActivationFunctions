#!/usr/bin python

import sys
import numpy as np
import os
from os import path

file_path = sys.argv[1]

# Find data start.
skip = 0
with open(file_path, 'r') as f:
	for l in f:
		skip += 1
		if '@data' in l:
			break
print('Skipping {} rows.'.format(skip))

# Load matrices.
features = np.loadtxt(file_path, delimiter=',', skiprows=skip)
labels = features[:, -1]
features = np.delete(features, -1, axis=1)

# Store matrices.
feat_file = path.join(path.dirname(file_path), 'traces_' + path.basename(file_path))
lab_file = path.join(path.dirname(file_path), 'labels_' + path.basename(file_path))

print(feat_file, lab_file)

np.savetxt(feat_file, features, delimiter=',')
np.savetxt(lab_file, labels, delimiter=',')

