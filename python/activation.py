import tensorflow as tf


class Tree:
    def __init__(self, string):
        self.root = self.__parse_tree(string)

    def build(self, input):
        return self.root.build(input)

    def __parse_tree(self, string):  # Recursive tree parsing
        i = string.find('[')
        if i == -1:  # leaf
            return parse_node(string)

        parts = string[i + 1: -1].split('[')  # +[sin[x],x] -> "+", "sin[x],x"
        node = parse_node(parts[0])

        parts = parts[1].split(',')  # Recursively construct children
        for p in parts:
            node.add(self.__parse_tree(p))
        return node


class Node:
    def __init__(self, build_method):
        self.children = []
        self.build_method = build_method

    def add(self, other):
        self.children.append(other)

    def build(self, input):
        return self.build_method(self.children, input)


VARIABLE_COUNT = 0


def parse_node(s):  # String to node
    if s.isdigit():  # Constant.
        return Node(lambda c, input: tf.constant(float(s)))

    if s == 'l':  # Learnable node
        VARIABLE_COUNT += 1
        return Node(lambda c, input: tf.get_variable("my_var" + str(VARIABLE_COUNT), shape=input.get_shape()[-1],
                                                     initializer=tf.constant_initializer(0.0), dtype=tf.float32))
    return nodes[s]


nodes = {
    ## Binary
    '+': Node(lambda c, input: tf.add(c[0].build(input), c[1].build(input))),
    '-': Node(lambda c, input: tf.subtract(c[0].build(input), c[1].build(input))),
    '*': Node(lambda c, input: tf.multiply(c[0].build(input), c[1].build(input))),
    '/': Node(lambda c, input: tf.divide(c[0].build(input), c[1].build(input))),
    'min': Node(lambda c, input: tf.minimum(c[0].build(input), c[1].build(input))),
    'max': Node(lambda c, input: tf.maximum(c[0].build(input), c[1].build(input))),
    'pow': Node(lambda c, input: tf.pow(c[0].build(input), c[1].build(input))),
    ## Unary
    'abs': Node(lambda c, input: tf.abs(c[0].build(input))),
    'neg': Node(lambda c, input: tf.negate(c[0].build(input))),
    'sin': Node(lambda c, input: tf.sin(c[0].build(input))),
    'cos': Node(lambda c, input: tf.cos(c[0].build(input))),
    'tan': Node(lambda c, input: tf.tan(c[0].build(input))),
    'exp': Node(lambda c, input: tf.exp(c[0].build(input))),
    'pow2': Node(lambda c, input: tf.square(c[0].build(input))),
    'pow3': Node(lambda c, input: tf.pow(c[0].build(input), tf.constant(3.))),
    'log': Node(lambda c, input: tf.log(c[0].build(input))),
    'gauss': Node(lambda c, input: tf.exp(tf.negate(tf.square(c[0].build(input))))),
    'sigmoid': Node(lambda c, input: tf.sigmoid(c[0].build(input))),
    'swish': Node(lambda c, input: tf.nn.swish(c[0].build(input))),
    # Relus
    'relu': Node(lambda c, input: tf.nn.relu(c[0].build(input))),
    'relu6': Node(lambda c, input: tf.nn.relu6(c[0].build(input))),  #
    'lrelu': Node(lambda c, input: tf.nn.leaky_relu(c[0].build(input))),
    'selu': Node(lambda c, input: tf.nn.selu(c[0].build(input))),
    'elu': Node(lambda c, input: tf.nn.elu(c[0].build(input))),
    'prelu': Node(lambda c, input: prelu(input)),
    # Softies
    'softmax': Node(lambda c, input: tf.nn.softmax(c[0].build(input))),  #
    'softplus': Node(lambda c, input: tf.nn.softplus(c[0].build(input))),
    'softsign': Node(lambda c, input: tf.nn.softsign(c[0].build(input))),
    # Hyperbolic
    'sinh': Node(lambda c, input: tf.sinh(c[0].build(input))),  #
    'cosh': Node(lambda c, input: tf.cosh(c[0].build(input))),  #
    'tanh': Node(lambda c, input: tf.tanh(c[0].build(input))),
    # Input
    'x': Node(lambda c, input: input)
}


def prelu(x):
    alpha = parse_node('l')
    pos = tf.nn.relu(x)
    neg = alpha * (x - tf.abs(x)) * 0.5
    return pos + neg
