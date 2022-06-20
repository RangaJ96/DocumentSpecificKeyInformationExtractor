import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class Adjacency(Layer):

    def __init__(self, n_features=50, max_nodes=50):
        super(Adjacency, self).__init__()
        self.max_nodes = max_nodes
        self.n_features = n_features
        self.input_units = max_nodes * max_nodes

    def build(self, input_shape=(50, 50)):
        self.w0_1 = self.add_weight(shape=(self.input_units, self.input_units),
                                    initializer='random_normal',
                                    trainable=True, name='w0_1')
        self.w0_2 = self.add_weight(shape=(self.input_units, self.input_units),
                                    initializer='random_normal',
                                    trainable=True, name='w0_2')

        self.w1_1 = self.add_weight(shape=(self.input_units, self.input_units),
                                    initializer='random_normal',
                                    trainable=True, name='w1_1')
        self.w1_2 = self.add_weight(shape=(self.input_units, self.input_units),
                                    initializer='random_normal',
                                    trainable=True, name='w1_2')

        self.w2_1 = self.add_weight(shape=(self.input_units, self.input_units),
                                    initializer='random_normal',
                                    trainable=True, name='w2_1')
        self.w2_2 = self.add_weight(shape=(self.input_units, self.input_units),
                                    initializer='random_normal',
                                    trainable=True, name='w2_2')

    def call(self, inputs):
        adj_list = inputs[:-1]
        node_vec = inputs[-1]

        assert len(adj_list) == 3, f'The `adj_list` passed to the layer does \
            not contain 3 adjacency operators. Received `adj_list` of length: \
                {len(adj_list)}.'

        try:
            assert adj_list[0].shape[0] == self.w0.shape[0], f'The number of rows \
                of the adjacency matrix and weight matrix does not match. /n\
                    Adjacency Shape: {adj_list.shape}\nWeight Matrix Shape: \
                        {self.w0.shape}.'

            assert adj_list[0].shape[0] == self.w0.shape[1], f'The number of \
                columns of the adjacency matrix and weight matrix does not match./n\
                    Adjacency Shape: {adj_list.shape}\nWeight Matrix Shape: \
                        {self.w0.shape}.'
        except:
            print("Error 87 layer")

        adj_0 = tf.squeeze(adj_list[0])
        adj_1 = tf.squeeze(adj_list[1])
        adj_2 = tf.squeeze(adj_list[2])
        node_vec = tf.squeeze(node_vec)

        shape = adj_0.shape

        adj_0 = self._learn_adjacencies(adj_0, node_vec)
        adj_1 = self._learn_adjacencies(adj_1, node_vec)
        adj_2 = self._learn_adjacencies(adj_2, node_vec)

        adj_0 = tf.reshape(adj_0, [-1])
        adj_1 = tf.reshape(adj_1, [-1])
        adj_2 = tf.reshape(adj_2, [-1])

        adj_0 = tf.matmul(adj_0 * self.w0_1)
        adj_0 = tf.nn.relu(adj_0)
        adj_0 = tf.matmul(adj_0 * self.w0_2)
        adj_0 = tf.nn.relu(adj_0)

        adj_1 = tf.matmul(adj_1 * self.w1_1)
        adj_1 = tf.nn.relu(adj_1)
        adj_1 = tf.matmul(adj_1 * self.w1_2)
        adj_1 = tf.nn.relu(adj_1)

        adj_2 = tf.matmul(adj_2 * self.w2_1)
        adj_2 = tf.nn.relu(adj_2)
        adj_2 = tf.matmul(adj_2 * self.w2_2)
        adj_2 = tf.nn.relu(adj_2)

        adj_0 = tf.reshape(adj_0, shape)
        adj_1 = tf.reshape(adj_1, shape)
        adj_2 = tf.reshape(adj_2, shape)

        return [adj_0, adj_1, adj_2]

    def _learn_adjacencies(self, adj, node_vec):

        new_adj = tf.zeros_like(adj, dtype=tf.float32)

        for ik, i in enumerate(adj):

            for jk, j in enumerate(i):
                adj_ij = j

                if adj_ij == 0:
                    new_adj[ik, jk] = 0

                elif adj_ij == 1:
                    new_adj[ik, jk] = tf.norm(node_vec[ik] - node_vec[jk])

        return new_adj

    def compute_output_shape(self, input_shape=[(1, 50, 50), (1, 50, 50),
                                                (1, 50, 50), (1, 50, 50)]):
        return input_shape[:-1]


class GNN(Layer):

    def __init__(self, n_features=50, n_nodes=50):
        super(GNN, self).__init__()
        self.n_features = n_features
        self.n_nodes = n_nodes

    def build(self, input_shape=None):

        self.w0 = self.add_weight(shape=(self.n_features, self.n_features),
                                  initializer='random_normal',
                                  trainable=True, name='w0')

        self.w1 = self.add_weight(shape=(self.n_features, self.n_features),
                                  initializer='random_normal',
                                  trainable=True, name='w1')

        self.w2 = self.add_weight(shape=(self.n_features, self.n_features),
                                  initializer='random_normal',
                                  trainable=True, name='w2')

    def call(self, inputs):

        X, learned_A0, learned_A1, learned_A2 = inputs

        product_1 = tf.matmul(tf.matmul(learned_A0, X), self.w0)
        product_2 = tf.matmul(tf.matmul(learned_A1, X), self.w1)
        product_3 = tf.matmul(tf.matmul(learned_A2, X), self.w2)

        X = tf.math.add(tf.math.add(product_1, product_2), product_3)
        X = tf.nn.relu(X)

        return X

    def compute_output_shape(self, input_shape=[(1, 50, 50), (1, 50, 50),
                                                (1, 50, 50), (1, 50, 50)]):
        return input_shape[0]


class GraphOperator(Layer):

    def __init__(self, power=2):
        super(GraphOperator, self).__init__()
        self.power = power

    def call(self, adj):

        assert adj.shape[1] == adj.shape[2], 'Adjacency matrix is not square. \
            Received adjacency matrix with shape {}'.format(adj.shape)

        adj = tf.squeeze(adj)

        n = adj.shape[0]

        A0 = tf.eye(n)
        A1 = tf.matmul(adj, adj)
        A2 = tf.matmul(A1, adj)

        return [A0, A1, A2]

    def compute_output_shape(self, input_shape=(50, 50)):
        return [input_shape, input_shape, input_shape]
