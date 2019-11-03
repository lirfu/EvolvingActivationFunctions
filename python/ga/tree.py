import tensorflow as tf
import re
import math as m


class Tree:
    def __init__(self, string):
        string = string.strip()
        if string.find('[') < 0:  # Leaf root
            self.root = parse_node(string).clone()
        else:
            self.root = None
            lst = re.split('[\[,\]]+', string)[:-1]
            for l in lst:
                if not self.root:
                    self.root = parse_node(l)
                    continue
                self.__add_node(self.root, parse_node(l))

    def get(self, index):
        i = [index]
        return self.root.get(i)

    def build(self, input):
        return self.root.build(input)

    def update(self, session):
        self.root.update(session)

    def size(self):
        return self.root.size()

    def __add_node(self, parent, node):
        for i in range(parent.children_num):
            if not parent.children[i]:
                parent.children[i] = node
                return True
            if self.__add_node(parent.children[i], node):  # Recurse into children.
                return True
        return False

    def to_string(self):
        return self.root.to_string()


class Node:
    def __init__(self, name, children_num, build_method):
        self.name = name
        self.children_num = children_num
        self.children = [None]*children_num
        self.build_method = build_method

    def clone(self):
        n = Node(self.name, self.children_num, self.build_method)
        n.children = []
        for c in self.children:
            if c:
                n.children.append(c.clone())
            else:
                n.children.append(None)
        return n

    def get(self, index):
        if index[0] == 0:
            return self
        index[0] -= 1
        for c in self.children:
            v = c.get(index)
            if v:
                return v
        return None

    def swap(self, node):
        self.name, node.name = node.name, self.name
        self.children_num, node.children_num = node.children_num, self.children_num
        self.children, node.children = node.children, self.children
        self.build_method, node.build_method = node.build_method, self.build_method

    def build(self, input):
        return self.build_method(self.children, input)

    def update(self, session):
        for c in self.children:
            c.update(session)

    def size(self):
        s = 1
        for c in self.children:
            s += c.size()
        return s

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

    def swap(self, node):
        super().swap(node)
        self.value, node.value = node.value, self.value

    def to_string(self):
        return str(self.value)

class LearnableNode(Node):
    VARIABLE_INDEX = 0

    def __init__(self, value):
        self.var_index = LearnableNode.VARIABLE_INDEX
        self.value = value

        super().__init__('l'+str(value), 0, lambda c, input: tf.get_variable('acti_var' + str(self.var_index), shape=input.get_shape()[-1],
                                         initializer=tf.constant_initializer(value), dtype=tf.float32))
        LearnableNode.VARIABLE_INDEX += 1

    def swap(self, node):
        super().swap(node)
        self.var_index, node.var_index = node.var_index, self.var_index
        self.value, node.value = node.value, self.value

    def update(self, session):
        self.value = tf.contrib.framework.get_variables('acti_var'+str(self.var_index))[0].eval(session=session)

    def to_string(self):
        return 'l'+str(self.value)


def parse_node(s):  # String to node
    try:
        float(s)  # Constant.
        return ConstNode(float(s))
    except ValueError:
        pass
    if re.match('l[-]*[0-9.]+', s):  # Learnable node
        try:
            float(s[1:])  # Constant.
            return LearnableNode(float(s[1:]))
        except ValueError:
            return None
    if s in nodes:
        return nodes[s].clone()
    else:
        raise KeyError('Unknown key: '+s)

tf_nodes = {
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

def relu(x):
    return max(0, x)

def relu6(x):
    return min(6, relu(x))

def lrelu(x):
    return x if x > 0 else 0.1 * x

def threlu(x):
    return x if x > 1 else 0

def elu(x):
    return x if x > 0 else m.exp(x) - 1

def selu(x):
    l = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return l*x if x > 0 else l*alpha * (m.exp(x) - 1)

def sigmoid(x):
    return 1. / (1 + m.exp(-x))

def h_sigmoid(x):
    return min(1, max(0, 0.2 * x + 0.5))

def swish(x):
    return x * sigmoid(x)

def elish(x):
    if x >= 0:
        return swish(x)
    else:
        return elu(x) * sigmoid(x)

def h_elish(x):
    a = lambda t: max(0, min(1, (t + 1) / 2))
    if x >= 0:
        return x * a(x)
    else:
        return elu(x) * a(x)

def tanh(x):
    e = m.exp(2 * x)
    return (e - 1) / (e + 1)

def h_tanh(x):
    if x < -1:
        return -1
    elif x > 1:
        return 1
    return x

def rat_tanh(x):
    dis= x * 2/3.
    s = signum(dis) * (1 - 1./(1 + abs(dis) + dis**2 + 1.41645 * dis**4))
    return 1.7159 * s

def rec_tanh(x):
    return max(0, tanh(x))

def softplus(x):
    return m.log(1+m.exp(x))

def softsign(x):
    return x / (1+abs(x))

def sin(x):
    return m.sin(x)

def cos(x):
    return m.cos(x)

def tr_sin(x):
    if x >= -m.pi/2 and x<= m.pi/2:
        return m.sin(x)
    elif x > m.pi/2:
        return 1
    return -1

def tr_cos(x):
    if x >= -m.pi/2 and x<= m.pi/2:
        return m.cos(x)
    return 0

def pow2(x):
    return x*x

def pow3(x):
    return x*x*x

def gauss(x):
    return m.exp(-x**2)

def softmax(x):
    return m.exp(x) / sum(m.exp(x))

py_nodes = {
    ## Binary
    '+': Node('+', 2, lambda c, input: c[0].build(input) + c[1].build(input)),
    '-': Node('-', 2, lambda c, input: c[0].build(input) - c[1].build(input)),
    '*': Node('*', 2, lambda c, input: c[0].build(input) * c[1].build(input)),
    '/': Node('/', 2, lambda c, input: c[0].build(input) / c[1].build(input)),
    'min': Node('min', 2, lambda c, input: min(c[0].build(input), c[1].build(input))),
    'max': Node('max', 2, lambda c, input: max(c[0].build(input), c[1].build(input))),
    'pow': Node('pow', 2, lambda c, input: m.pow(c[0].build(input), c[1].build(input))),
    ## Unary
    'abs': Node('abs', 1, lambda c, input: abs(c[0].build(input))),
    'neg': Node('neg', 1, lambda c, input: -c[0].build(input)),
    'sin': Node('sin', 1, lambda c, input: m.sin(c[0].build(input))),
    'cos': Node('cos', 1, lambda c, input: m.cos(c[0].build(input))),
    'tan': Node('tan', 1, lambda c, input: m.tan(c[0].build(input))),
    'exp': Node('exp', 1, lambda c, input: m.exp(c[0].build(input))),
    'pow2': Node('pow2', 1, lambda c, input: m.pow(c[0].build(input), 2)),
    'pow3': Node('pow3', 1, lambda c, input: m.pow(c[0].build(input), 3)),
    'log': Node('log', 1, lambda c, input: m.log(c[0].build(input))),
    'gauss': Node('gauss', 1, lambda c, input: gauss(c[0].build(input))),
    'sigmoid': Node('sigmoid', 1, lambda c, input: sigmoid(c[0].build(input))),
    'swish': Node('swish', 1, lambda c, input: swish(c[0].build(input))),
    # Relus
    'relu': Node('relu', 1, lambda c, input: relu(c[0].build(input))),
    'relu6': Node('relu6', 1, lambda c, input: relu6(c[0].build(input))),
    'lrelu': Node('lrelu', 1, lambda c, input: lrelu(c[0].build(input)),
    'selu': Node('selu', 1, lambda c, input: selu(c[0].build(input))),
    'elu': Node('elu', 1, lambda c, input: elu(c[0].build(input))),
    #'prelu': Node('prelu', 1, lambda c, input: prelu(input)),
    # Softies
    'softmax': Node('softmax', 1, lambda c, input: softmax(c[0].build(input)),
    'softplus': Node('softplus', 1, lambda c, input: softplus(c[0].build(input))),
    'softsign': Node('softsign', 1, lambda c, input: softsign(c[0].build(input))),
    # Hyperbolic
    'sinh': Node('sinh', 1, lambda c, input: tf.sinh(c[0].build(input))),
    'cosh': Node('cosh', 1, lambda c, input: tf.cosh(c[0].build(input))),
    'tanh': Node('tanh', 1, lambda c, input: tf.tanh(c[0].build(input))),
    # Input
    'x': Node('x', 0, lambda c, input: input)
}


def prelu(x):
    alpha = parse_node('l0')
    pos = tf.nn.relu(x)
    neg = alpha * (x - tf.abs(x)) * 0.5
    return pos + neg

nodes = tf_nodes

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # Define target function.
    func_target = lambda x: np.exp(-(x - 3.14) ** 2)
    # Define model function.
    func_model = 'relu[-[1,abs[-[x,l1.0]]]]'

    # Generate data.
    x = np.linspace(-5, 5, 300).reshape([-1, 1])
    y = np.array([func_target(t) for t in x]).reshape([-1, 1])

    # Build tree.
    tree = Tree(func_model)
    tf_truth = tf.placeholder(tf.float32, [None, 1])
    tf_inp = tf.placeholder(tf.float32, [None, 1])
    tf_out = tree.build(tf_inp)

    tf_loss = tf.reduce_sum((tf_truth - tf_out) ** 2)
    tf_optimi = tf.train.GradientDescentOptimizer(0.01).minimize(tf_loss)
    tf_sess = tf.Session()
    tf_sess.run(tf.initialize_all_variables())

    # Train learnable nodes.
    ls_prev = -1
    for i in range(1,101):
        ls, _ = tf_sess.run([tf_loss, tf_optimi], {tf_inp: x, tf_truth: y})
        print('Iter', i, 'has loss:', ls)
        if abs(ls_prev-ls) < 1e-12:
            break
        ls_prev = ls
    tree.update(tf_sess)

    # Collect final predictions.
    p = tf_sess.run([tf_out], {tf_inp: x})
    p = np.array(p).reshape(-1)

    # Plot.
    plt.figure(figsize=(10,4))
    plt.plot(x, y, 'b', label='Original function')
    plt.plot(x, p, 'r', label=tree.to_string())
    plt.legend()
    plt.show()
