#!/usr/bin python

import sys, os
import json
import numpy as np

if len(sys.argv) != 2:
	print("Please provide a predictions file!")
	exit(1)

with open(sys.argv[1], 'r') as f:
	js = json.load(f)
	shape = tuple(js['shape'])
	data = np.array(js['data'])
	if shape != tuple(data.shape):
		print('Loaded array shape missmatch:')
		print('--> File:', shape)
		print('--> Loaded:', tuple(data.shape))

pred = np.argmax(data, axis=1)

with open(sys.argv[1]+'.np', 'wb') as f:
	np.save(f, pred)
