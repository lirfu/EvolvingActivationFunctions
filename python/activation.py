import tensorflow as tf
import re


class Tree:
    def __init__(self, string):
        self._parse_tree(string)

    def build(self, input):
        return self.root.build(input)

    def __add_node(self, parent, node):
        for i in range(parent.children_num):
            if not parent.children[i]:
                parent.children[i] = node
                return True
            if self.__add_node(parent.children[i], node):  # Recurse into children.
                return True

    def _parse_tree(self, string):  # Recursive tree parsing
        string = string.strip()
        if string.find('[') < 0:  # Leaf root
            self.root = parse_node(string)
        else:
            self.root = None
            list = re.split('[\[,\]]+', string)[:-1]
            for l in list:
                if not self.root:
                    self.root = parse_node(l)
                    continue
                self.__add_node(self.root, parse_node(l))

    def to_string(self):
        return self.root.to_string()


class Node:
    def __init__(self, name, children_num, build_method):
        self.name = name
        self.children_num = children_num
        self.children = [None]*children_num
        self.build_method = build_method

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

class ConstNode(Node):
    def __init__(self, value):
        super().__init__("const", 0, lambda c, input: tf.constant(value))
        self.value = value

    def to_string(self):
        return str(self.value)

class LearnableNode(Node):
    VARIABLE_INDEX = 0

    def __init__(self, value):
        self.var_index = LearnableNode.VARIABLE_INDEX
        LearnableNode.VARIABLE_INDEX += 1

        super().__init__('l'+str(value), 0, lambda c, input: tf.get_variable("acti_var" + str(self.var_index), shape=input.get_shape()[-1],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32))
        self.value = value

    def to_string(self):
        return 'l'+str(self.value)


def parse_node(s):  # String to node
    try:
        float(s)  # Constant.
        return ConstNode(float(s))
    except ValueError:
        pass
    if re.match('l[0-9.]+', s):  # Learnable node
        try:
            float(s[1:])  # Constant.
            return LearnableNode(float(s[1:]))
        except ValueError:
            pass
    return nodes[s]


nodes = {
    ## Binary
    '+': Node('+', 2, lambda c, input: tf.add(c[0].build(input), c[1].build(input))),
    '-': Node('-', 2, lambda c, input: tf.subtract(c[0].build(input), c[1].build(input))),
    '*': Node('*', 2, lambda c, input: tf.multiply(c[0].build(input), c[1].build(input))),
    '/': Node('/', 2, lambda c, input: tf.divide(c[0].build(input), c[1].build(input))),
    'min': Node('min', 2, lambda c, input: tf.minimum(c[0].build(input), c[1].build(input))),
    'max': Node('max', 2, lambda c, input: tf.maximum(c[0].build(input), c[1].build(input))),
    'pow': Node('pow', 2, lambda c, input: tf.pow(c[0].build(input), c[1].build(input))),
    ## Unary
    'abs': Node('abs', 1, lambda c, input: tf.abs(c[0].build(input))),
    'neg': Node('neg', 1, lambda c, input: tf.negate(c[0].build(input))),
    'sin': Node('sin', 1, lambda c, input: tf.sin(c[0].build(input))),
    'cos': Node('cos', 1, lambda c, input: tf.cos(c[0].build(input))),
    'tan': Node('tan', 1, lambda c, input: tf.tan(c[0].build(input))),
    'exp': Node('exp', 1, lambda c, input: tf.exp(c[0].build(input))),
    'pow2': Node('pow2', 1, lambda c, input: tf.square(c[0].build(input))),
    'pow3': Node('pow3', 1, lambda c, input: tf.pow(c[0].build(input), tf.constant(3.))),
    'log': Node('log', 1, lambda c, input: tf.log(c[0].build(input))),
    'gauss': Node('gauss', 1, lambda c, input: tf.exp(tf.negate(tf.square(c[0].build(input))))),
    'sigmoid': Node('sigmoid', 1, lambda c, input: tf.sigmoid(c[0].build(input))),
    'swish': Node('swish', 1, lambda c, input: tf.nn.swish(c[0].build(input))),
    # Relus
    'relu': Node('relu', 1, lambda c, input: tf.nn.relu(c[0].build(input))),
    'relu6': Node('relu6', 1, lambda c, input: tf.nn.relu6(c[0].build(input))),  #
    'lrelu': Node('lrelu', 1, lambda c, input: tf.nn.leaky_relu(c[0].build(input))),
    'selu': Node('selu', 1, lambda c, input: tf.nn.selu(c[0].build(input))),
    'elu': Node('elu', 1, lambda c, input: tf.nn.elu(c[0].build(input))),
    'prelu': Node('prelu', 1, lambda c, input: prelu(input)),
    # Softies
    'softmax': Node('softmax', 1, lambda c, input: tf.nn.softmax(c[0].build(input))),  #
    'softplus': Node('softplus', 1, lambda c, input: tf.nn.softplus(c[0].build(input))),
    'softsign': Node('softsign', 1, lambda c, input: tf.nn.softsign(c[0].build(input))),
    # Hyperbolic
    'sinh': Node('sinh', 1, lambda c, input: tf.sinh(c[0].build(input))),  #
    'cosh': Node('cosh', 1, lambda c, input: tf.cosh(c[0].build(input))),  #
    'tanh': Node('tanh', 1, lambda c, input: tf.tanh(c[0].build(input))),
    # Input
    'x': Node('x', 0, lambda c, input: input)
}


def prelu(x):
    alpha = parse_node('l')
    pos = tf.nn.relu(x)
    neg = alpha * (x - tf.abs(x)) * 0.5
    return pos + neg


if __name__ == '__main__':
    s = "+[relu[x],tanh[*[l2.69,3.14]]]"
    # s = "+[12.3,sin[l5.3]]"
    print("Before:", s)
    t = Tree(s)
    print("After: ", t.to_string())
