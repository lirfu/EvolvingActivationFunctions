import tensorflow
from tensorflow.contrib import layers

class LayerDescriptor:
    def __init__(self, size):
        self.size = size

    def construct(self, input):
        raise Exception("Not implemented!")

class FCDescriptor(LayerDescriptor):
    def construct(self, input):
        return layers.fully_connected(input, self.size, activation_fn=None)
