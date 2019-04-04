import tensorflow as tf
import re

class Tree:
    def __init__(self, string):
        self.root = self._parse_tree(string)

    def build(self, input):
        return self.root.build(input)

    def _parse_tree(self, string):  # Recursive tree parsing
        string = string.strip()
        print(string)

        nodes = []

        i = string.find('[')
        if i < 0:
            return parse_node(string)

        n = parse_node(string[:i])

        i = string.find('[')
        j = string.find(']')
        k = string.find(',')
        m = min(i,j,k)

        if m == i:
            nodes.append(n)
            n = parse_node(string[:i])
            string = string[i+1:]

        elif m == j:
            n = nodes.pop()

        elif m == k:


    def to_string(self):
        return self.root.to_string()


class Node:
    def __init__(self, name, build_method):
        self.name = name
        self.children = []
        self.build_method = build_method

    def add(self, other):
        self.children.append(other)

    def build(self, input):
        return self.build_method(self.children, input)

    def to_string(self):
        if len(self.children) == 0:
            return self.name

        s = self.name + "["
        t = False
        for c in self.children:
            if t:
                s += ','
            s += c.to_string()
            t = True
        return s + "]"


VARIABLE_COUNT = 0


def parse_node(s):  # String to node
    try:
        float(s)  # Constant.
        return Node('const', lambda c, input: tf.constant(float(s)))
    except ValueError:
        pass

    if s == 'l':  # Learnable node
        VARIABLE_COUNT += 1
        return Node('l', lambda c, input: tf.get_variable("my_var" + str(VARIABLE_COUNT), shape=input.get_shape()[-1],
                                                     initializer=tf.constant_initializer(0.0), dtype=tf.float32))
    return nodes[s]


nodes = {
    ## Binary
    '+': Node('+', lambda c, input: tf.add(c[0].build(input), c[1].build(input))),
    '-': Node('-',lambda c, input: tf.subtract(c[0].build(input), c[1].build(input))),
    '*': Node('*',lambda c, input: tf.multiply(c[0].build(input), c[1].build(input))),
    '/': Node('/',lambda c, input: tf.divide(c[0].build(input), c[1].build(input))),
    'min': Node('min',lambda c, input: tf.minimum(c[0].build(input), c[1].build(input))),
    'max': Node('max',lambda c, input: tf.maximum(c[0].build(input), c[1].build(input))),
    'pow': Node('pow',lambda c, input: tf.pow(c[0].build(input), c[1].build(input))),
    ## Unary
    'abs': Node('abs',lambda c, input: tf.abs(c[0].build(input))),
    'neg': Node('neg',lambda c, input: tf.negate(c[0].build(input))),
    'sin': Node('sin',lambda c, input: tf.sin(c[0].build(input))),
    'cos': Node('cos',lambda c, input: tf.cos(c[0].build(input))),
    'tan': Node('tan',lambda c, input: tf.tan(c[0].build(input))),
    'exp': Node('exp',lambda c, input: tf.exp(c[0].build(input))),
    'pow2': Node('pow2',lambda c, input: tf.square(c[0].build(input))),
    'pow3': Node('pow3',lambda c, input: tf.pow(c[0].build(input), tf.constant(3.))),
    'log': Node('log',lambda c, input: tf.log(c[0].build(input))),
    'gauss': Node('gauss',lambda c, input: tf.exp(tf.negate(tf.square(c[0].build(input))))),
    'sigmoid': Node('sigmoid',lambda c, input: tf.sigmoid(c[0].build(input))),
    'swish': Node('swish',lambda c, input: tf.nn.swish(c[0].build(input))),
    # Relus
    'relu': Node('relu',lambda c, input: tf.nn.relu(c[0].build(input))),
    'relu6': Node('relu6',lambda c, input: tf.nn.relu6(c[0].build(input))),  #
    'lrelu': Node('lrelu',lambda c, input: tf.nn.leaky_relu(c[0].build(input))),
    'selu': Node('selu',lambda c, input: tf.nn.selu(c[0].build(input))),
    'elu': Node('elu',lambda c, input: tf.nn.elu(c[0].build(input))),
    'prelu': Node('prelu',lambda c, input: prelu(input)),
    # Softies
    'softmax': Node('softmax',lambda c, input: tf.nn.softmax(c[0].build(input))),  #
    'softplus': Node('softplus',lambda c, input: tf.nn.softplus(c[0].build(input))),
    'softsign': Node('softsign',lambda c, input: tf.nn.softsign(c[0].build(input))),
    # Hyperbolic
    'sinh': Node('sinh',lambda c, input: tf.sinh(c[0].build(input))),  #
    'cosh': Node('cosh',lambda c, input: tf.cosh(c[0].build(input))),  #
    'tanh': Node('tanh',lambda c, input: tf.tanh(c[0].build(input))),
    # Input
    'x': Node('x',lambda c, input: input)
}


def prelu(x):
    alpha = parse_node('l')
    pos = tf.nn.relu(x)
    neg = alpha * (x - tf.abs(x)) * 0.5
    return pos + neg

if __name__ == '__main__':
    s = "+[relu[x],tanh[*[l,3.14]]]"
    print("Before:",s)
    t = Tree(s)
    print("After:",t.to_string())
