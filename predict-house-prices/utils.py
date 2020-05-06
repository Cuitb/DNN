from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def hidden_layer(layer_input, output_depth, scope='hidden_layer', reuse=None):

    input_depth = layer_input.get_shape()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable(initializer=tf.random_normal_initializer(), shape=(input_depth, output_depth), name='weights')
        b = tf.get_variable(initializer=tf.zeros_initializer(), shape=(output_depth),name='biases')
        net = tf.matmul(layer_input, w) + b
        return net

def DNN(x, output_depths, scope='DNN', reuse=None):
    net = x
    for i, output_depth in enumerate(output_depths):
        net = hidden_layer(net, output_depth, scope='layer%d'%i, reuse=reuse)
        net = tf.tanh(net)
    net = hidden_layer(net, 1, scope='classification', reuse=reuse)
    net = tf.sigmoid(net)
    return net



