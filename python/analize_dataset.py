import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split


def draw_hist(x, vals, label='', color='b'):
    plt.hist([l-0.5 for l in x], bins=vals, range=[-0.5,vals-0.5], facecolor=color, histtype=u'step', label=label, normed=False)


# Check args.
if len(sys.argv) < 2:
    if len(sys.argv)==1:
        print("Please provide the dataset path!")
#    elif len(sys.argv)==2:
#        print("Please provide the train percentage!")
    print("Usage: split_dataset.py <dataset_path>")
    exit(1)

# Get args.
dataset_path = sys.argv[1]
#train_percentage = float(sys.argv[2])

# Load data.
instances = []
d = False
with open(dataset_path, 'r') as f:
    r = csv.reader(f, delimiter=',')
    for line in r:
        if len(line)==0: # Skip empty.
            continue
        elif d: # Add instances.
            instances.append([float(i) for i in line])
        elif line[0] == "@data": # Start adding instances.
            d = True
instances = np.array(instances)

# Construct arrays.
inputs = np.array(instances[:, :-1])
labels = np.array(instances[:, -1])

plt.figure(figsize=(18,8))

# Draw original distribution.
ctr = Counter(labels)
draw_hist(labels, len(ctr), 'Original distribution', 'b')

# Split dataset.
#x_train, x_test, y_train, y_test = train_test_split(inputs, labels, train_size=train_percentage, random_state=42)

#print("Train", Counter(y_train))
#print("Test", Counter(y_train))

# Draw train distribution.
#draw_hist(y_train, len(ctr), 'Train distribution', 'g')

# Draw test distribution.
#draw_hist(y_test, len(ctr), 'Test distribution', 'r')

# Show graphs.
plt.xlabel('Labels')
plt.ylabel('Probability')
plt.title('Distribution of '+os.path.basename(dataset_path))
plt.legend()

img_file = os.path.splitext(dataset_path)[0] + '.png'
print('Saving image file: ' + img_file)
plt.savefig(img_file, format='png')
plt.show()
