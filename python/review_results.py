#!/usr/bin python

import os
import os.path as path
import sys
import re

arg_i = 1

if len(sys.argv) > 1:  # Arguments exist
	root_dir = sys.argv[arg_i]
	arg_i += 1
	if not path.exists(root_dir):
		print('Path not found:', root_dir)
		exit(1)
	if not path.isdir(root_dir):
		root_dir = path.dirname(root_dir)
		print('Path was a file, using its\' parent:', root_dir)
else:  # Use current dir
	root_dir = os.getcwd()

experiments = os.listdir(root_dir)
experiments.sort()

if len(sys.argv) > 2 or (len(sys.argv) > 1 and arg_i == 1):  # Filter out those without pattern
	pattern = sys.argv[arg_i]
	arg_i += 1
	experiments = [x for x in experiments if re.match(pattern, x)]

l_i = 2
u_i = 7
for e in experiments:
	print('Experiment:', e)
	print('Accuracy | Precision | Recall | F1 | AUC | AUC prob.')
	variants = os.listdir(path.join(root_dir, e))
	variants.sort()

	for v in variants:
#		print('--> Variant:', v)
		res = path.join(root_dir, e, v, 'results.txt')
		with open(res, 'r') as f:
			i = 0
			for l in f:
				if i < l_i or i > u_i:
					i += 1
					continue
				if i > l_i:
					print(' | ', sep='', end='', flush=False)
				print('{:.3f}'.format(round(float(l.split('\t')[1].strip()), 3)), sep='', end='', flush=False)
				i += 1
		print('')
