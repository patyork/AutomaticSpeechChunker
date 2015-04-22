__author__ = 'pat'
'''
Bidirectional Recurrent Neural Network
with Connectionist Temporal Classification (CTC)
  courtesy of https://github.com/shawntan/rnn-experiment
  courtesy of https://github.com/rakeshvar/rnn_ctc
implemented in Theano and optimized for use on a GPU
'''

import theano
import theano.tensor as T
from theano_toolkit import utils as U
from theano_toolkit import updates
import numpy as np
import cPickle as pickle
import time

#THEANO_FLAGS='device=cpu,floatX=float32'
#theano.config.warn_float64='warn'

#theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity='high'
theano.config.on_unused_input='warn'


class FeedForwardLayer:
    def __init__(self, inputs, input_size, output_size, rng, dropout_rate, parameters=None):
        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        if parameters is None:
            self.W = U.create_shared(U.initial_weights(input_size, output_size), name='W')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W = theano.shared(parameters['W'], name='W')
            self.b = theano.shared(parameters['b'], name='b')


        self.output = T.cast(self.activation_fn( (T.dot(inputs, self.W) + self.b)*(1.0-dropout_rate) ), dtype=theano.config.floatX)

        self.params = [self.W, self.b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def set_parameters(self, parameters):
        self.W.set_value(parameters['W'])
        self.b.set_value(parameters['b'])


class RecurrentLayer:
    def __init__(self, inputs, input_size, output_size, is_backward=False, parameters=None):

        if parameters is None:
            self.W_if = U.create_shared(U.initial_weights(input_size, output_size), name='W_if')
            self.W_ff = U.create_shared(U.initial_weights(output_size, output_size), name='W_ff')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W_if = theano.shared(parameters['W_if'], name='W_if')
            self.W_ff = theano.shared(parameters['W_ff'], name='W_ff')
            self.b = theano.shared(parameters['b'], name='b')

        initial = T.zeros((output_size,))
        self.is_backward = is_backward
        self.activation_fn = lambda x: T.cast(T.minimum(x * (x > 0), 20), dtype='float32')#dtype=theano.config.floatX)
        
        nonrecurrent = T.dot(inputs, self.W_if) + self.b

        self.output, _ = theano.scan(
            lambda in_t, out_tminus1, weights: self.activation_fn(in_t + T.dot(out_tminus1, weights)),
            sequences=[nonrecurrent],
            outputs_info=[initial],
            non_sequences=[self.W_ff],
            go_backwards=self.is_backward
        )

        self.params = [self.W_if, self.W_ff, self.b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def set_parameters(self, parameters):
        self.W_if.set_value(parameters['W_if'])
        self.W_ff.set_value(parameters['W_ff'])
        self.b.set_value(parameters['b'])


class SoftmaxLayer:
    def __init__(self, inputs, input_size, output_size, parameters=None):

        if parameters is None:
            self.W = U.create_shared(U.initial_weights(input_size, output_size), name='W')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W = theano.shared(parameters['W'], name='W')
            self.b = theano.shared(parameters['b'], name='b')

        self.output = T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        self.params = [self.W, self.b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def set_parameters(self, parameters):
        self.W.set_value(parameters['W'])
        self.b.set_value(parameters['b'])



class BRNN:
    def __init__(self, input_dimensionality, output_dimensionality, params=None, learning_rate=0.0001, momentum=.25):
        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality
        self.learning_rate = learning_rate
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed=1234)

        input_seq = T.fmatrix('input_seq')
        dropoutRate = T.fscalar('dropoutRate')

        if params is None:
            self.ff1 = FeedForwardLayer(input_seq, self.input_dimensionality, 2000, rng=srng, dropout_rate=dropoutRate)
            self.ff2 = FeedForwardLayer(self.ff1.output, 2000, 1000, rng=srng, dropout_rate=dropoutRate)
            self.ff3 = FeedForwardLayer(self.ff2.output, 1000, 800, rng=srng, dropout_rate=dropoutRate)
            self.rf = RecurrentLayer(self.ff3.output, 800, 500, False)     # Forward layer
            self.rb = RecurrentLayer(self.ff3.output, 800, 500, True)      # Backward layer

            # REVERSE THE BACKWARDS RECURRENT OUTPUTS IN TIME (from [T-1, 0] ===> [0, T-1]
            self.s = SoftmaxLayer(T.concatenate((self.rf.output, self.rb.output[::-1, :]), axis=1), 2*500, self.output_dimensionality)

        else:
            self.ff1 = FeedForwardLayer(input_seq, self.input_dimensionality, 2000, parameters=params[0], rng=srng, dropout_rate=dropoutRate)
            self.ff2 = FeedForwardLayer(self.ff1.output, 2000, 1000, parameters=params[1], rng=srng, dropout_rate=dropoutRate)
            self.ff3 = FeedForwardLayer(self.ff2.output, 1000, 800, parameters=params[2], rng=srng, dropout_rate=dropoutRate)
            self.rf = RecurrentLayer(self.ff3.output, 800, 500, False, parameters=params[3])     # Forward layer
            self.rb = RecurrentLayer(self.ff3.output, 800, 500, True, parameters=params[4])      # Backward layer

            # REVERSE THE BACKWARDS RECURRENT OUTPUTS IN TIME (from [T-1, 0] ===> [0, T-1]
            self.s = SoftmaxLayer(T.concatenate((self.rf.output, self.rb.output[::-1, :]), axis=1), 2*500, self.output_dimensionality, parameters=params[5])


        self.probabilities = theano.function(
            inputs=[input_seq, dropoutRate],
            outputs=[self.s.output],
            allow_input_downcast=True
        )


    def dump(self, f_path):
        f = file(f_path, 'wb')
        for obj in [self.ff1.get_parameters(), self.ff2.get_parameters(), self.ff3.get_parameters(), self.rf.get_parameters(), self.rb.get_parameters(), self.s.get_parameters()]:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

class Network:
    def __init__(self):
        self.nn = None

    def create_network(self, input_dimensionality, output_dimensionality, learning_rate=0.01, momentum=.25):
        self.nn = BRNN(input_dimensionality, output_dimensionality, params=None, learning_rate=learning_rate, momentum=momentum)
        return self.nn

    def load_network(self, path, input_dimensionality, output_dimensionality, learning_rate=0.00001, momentum=.75):
        f = file(path, 'rb')
        parameters = []
        for i in np.arange(6):
            parameters.append(pickle.load(f))
        f.close()

        self.nn = BRNN(input_dimensionality, output_dimensionality, params=parameters, learning_rate=learning_rate, momentum=momentum)
        return self.nn

    def dump_network(self, path):
        if self.nn is None:
            return False

        self.nn.dump(path)

    def get_network(self):
        assert(self.nn is not None)
        return [self.nn.ff1.get_parameters(), self.nn.ff2.get_parameters(), self.nn.ff3.get_parameters(), self.nn.rf.get_parameters(), self.nn.rb.get_parameters(), self.nn.s.get_parameters()]

    def set_network(self, parameters):
        assert(type(parameters) == list)
        assert(len(parameters) == 6)

        self.nn.ff1.set_parameters(parameters[0])
        self.nn.ff2.set_parameters(parameters[1])
        self.nn.ff3.set_parameters(parameters[2])
        self.nn.rf.set_parameters(parameters[3])
        self.nn.rb.set_parameters(parameters[4])
        self.nn.s.set_parameters(parameters[5])
        
    def set_network_from_file(self, fname):
        f = file(fname, 'rb')
        parameters = []
        for i in np.arange(6):
            parameters.append(pickle.load(f))
        f.close()
        
        self.set_network(parameters)
