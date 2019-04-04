import tensorflow as tf
from tensorflow.contrib import layers


class CommonModel:
    def __init__(self, params):
        self.params = params
        tf.reset_default_graph()
        tf.set_random_seed(params['seed'])

        # Define inputs.
        self.X = tf.placeholder(tf.float32, (None, params['input_size']))
        self.Y = tf.placeholder(tf.int32, (None))

        # Build network.
        with tf.contrib.framework.arg_scope([],
                                            weights_initializer=layers.variance_scaling_initializer(),
                                            weights_regularizer=layers.l2_regularizer(params['regularization_coef'])):
            layer = self.X
            for l in params['architecture']:
                layer = l.construct(layer)
                layer = params['activation'].build(layer)
                if params['batch_norm']:
                    layer = layers.batch_norm(layer)

        # Output is fully connected with softmax.
        out = layers.fully_connected(layer, params['output_size'], activation_fn=None)
        self.outputs = tf.argmax(out)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.Y, params['output_size']),
                                                               logits=self.out)
        if params['regularization_coef'] > 0:
            self.loss = tf.add(self.loss, params['regularization_coef'] *
                               tf.sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

        # Define learning rate decay.
        self.global_step = tf.Variable(0, trainable=False, name="GlobalStep")
        self.learning_rate = tf.train.exponential_decay(
            params['learning_rate'], self.global_step, params['decay_step'], params['decay_rate'], staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Build session.
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, features, labels, bridge):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.repeat(self.params['epochs_num'])
        dataset = dataset.batch(self.params['batch_size'])
        # TODO Implement model training

    def transform(self, features):
        dataset = tf.data.Dataset.from_tensors(features)
        dataset = dataset.batch(self.params['batch_size'])
        iterator = dataset.make_one_shot_iterator()
        next_X = iterator.get_next()

        predictions = []
        try:
            while True:
                predictions.append(self.sess.run([self.outputs, self.train_step], {self.X: next_X})) # TODO test this iterator thingy
        except tf.errors.OutOfRangeError:
            pass
        return predictions


    def score(self, features, labels, bridge):
        predictions = self.transform(features)
        pass # TODO Implement model scoring (metrics)


if __name__ == "__main__":
    params = {'input_size': 5, 'output_size': 5, 'regularization_coef': 0., 'learning_rate': 0.01, 'decay_step': 1,
              'decay_rate': 1, 'architecture': [FCDescriptor(10), FCDescriptor(10)], 'seed': 42, 'batch_norm': True}
    mod = CommonModel(params)
