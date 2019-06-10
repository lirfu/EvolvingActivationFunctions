#!/usr/bin python

import os
import os.path as path
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 3:
	print('Usage: heatmap.py [ROOT_DIR] [EXPERIMENT_PREFIX]')

root_dir = sys.argv[1]
exp_name = sys.argv[2]
#root_dir = 'noiseless_all_training_9class'
#exp_name = 'common_functions_v2'


# Experiment list
experiments = os.listdir(root_dir)
exp_data = {}
architectures = []
functions = []

for e in experiments:
	if not e.startswith(exp_name):
		continue	

	par = e.split(exp_name)[1]
	parts = par.split('_')
	
	arch = [int(s) for s_w in parts[1].split('-') for s in re.split('\(|\)', s_w) if s.isdigit()]
	func = parts[2].split('(')[0].split('[')[0]
	
	if str(arch) not in exp_data:
		exp_data[str(arch)] = []

	exp_data[str(arch)].append((e,func))

	if arch not in architectures:
		architectures.append(arch)
	if func not in functions:
		functions.append(func)

architectures.sort(key=lambda x: len(x) * 1000000 + sum(x))
for a in architectures:
	exp_data[str(a)].sort(key=lambda x: x[1])
functions.sort()

# Data extraction
hp_data = []
acc_data = []
f1_data = []
for a in architectures:
	hp_d = []
	a_d = []
	f1_d = []
	t = exp_data[str(a)]
	if t is None:
		print('No experiments for:', str(a))
		continue

	for e in t:
		fil = os.path.join(root_dir,e[0])
		best_dir = [f for f in os.listdir(fil) if f.endswith('BEST')][0]

		num = int(best_dir.split('_')[0])
		hp_d.append(num)

		fil = os.path.join(fil,best_dir,'results.txt')
		with open(fil) as f:
			for l in f:
				if l.startswith('accuracy'):
					a_d.append(float(l.split('\t')[1]))
				if l.startswith('f1\t'):
					f1_d.append(float(l.split('\t')[1]))

	hp_data.append(hp_d)
	acc_data.append(a_d)
	f1_data.append(f1_d)


# Drawing
def plt_heatmap(data, labx, laby, title, filesuffix):
	data = np.array(data).T

	fig = plt.figure(figsize=(15,10))
	ax = fig.add_subplot(1, 1, 1)

	im = ax.imshow(data)
	ax.figure.colorbar(im)

	ax.set_title(title)
	ax.set_xticks(np.arange(len(labx)))
	ax.set_yticks(np.arange(len(laby)))
	ax.set_xticklabels(labx)
	ax.set_yticklabels(laby)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#	heatmap = plt.pcolor(data)
#	plt.colorbar(heatmap)

	for y in range(data.shape[0]):
		for x in range(data.shape[1]):
		    plt.text(x, y, '%03d' % int(data[y,x]*1000) if data[y,x] < 1 else data[y,x],
					ha='center', va='center', fontsize=8, 
					color='white' if data[y,x]<0.5 else 'black')

	plt.tight_layout()
	plt.savefig(root_dir + '-' + exp_name + '_' + filesuffix + '.pdf', bbox_inches='tight')
	plt.show()


arch_labels = ['-'.join([str(i) for i in a]) for a in architectures]

#plt_heatmap(hp_data, arch_labels, functions, 'Hiperparametri', 'hp')
#plt_heatmap(acc_data, arch_labels, functions, 'Točnost', 'accuracy')
#plt_heatmap(f1_data, arch_labels, functions, 'F1 mjera', 'f1')

f1_data = np.sum(np.square(np.array(f1_data)), axis=1)
print('Best architecture:')
print(arch_labels)
print(f1_data)


plt.figure()
plt.bar(range(f1_data.shape[0]), f1_data)
plt.xticks(range(f1_data.shape[0]), arch_labels)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.savefig(root_dir + '-' + exp_name + '_f1squaredquality.pdf', bbox_inches='tight')
plt.show()


from collections import Counter
hp_data = np.array(hp_data)[2,:]
ctr = Counter(hp_data)
print('Best hyperparameters:', ctr)
