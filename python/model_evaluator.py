import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

from data import load, parse_params
from java_bridge import JavaBridge
from common_model import CommonModel

# Create the bridge.
bridge = JavaBridge()

# Read and parse the parameters.
ps = bridge.read()
params = parse_params(ps)

# Load data from file.
train_X, train_Y = load(params['train_path'])
test_X, test_Y = load(params['test_path'])

# Normalize datasets.
if params['normalize_features']:
    norm = Normalizer().fit(train_X)
    train_X = norm.transform(train_X, copy=False)
    test_X = norm.transform(test_X, copy=False)

# Train and score the model.
model = CommonModel(params)
model.fit(train_X, train_Y, bridge)
model.score(test_X, test_Y, bridge)

# Close the bridge and end program.
bridge.end()
