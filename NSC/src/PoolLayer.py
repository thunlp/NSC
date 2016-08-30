#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy

class LastPoolLayer(object):
    def __init__(self, input):
        self.input = input
        self.output = input[-1]
        self.params = []

    def save(self, prefix):
        pass

class MeanPoolLayer(object):
    def __init__(self, input, ll):
        self.input = input
        self.output = T.sum(input, axis=0, acc_dtype='float32') / ll.dimshuffle(0, 'x')          
        self.params = []

    def save(self, prefix):
        pass


class MaxPoolLayer(object):
    def __init__(self, input):
        self.input = input
        self.output = T.max(input, axis = 0)
        self.params = []

    def save(self, prefix):
        pass

class Dropout(object):
    def __init__(self, input, rate, istrain):
        rate = numpy.float32(rate)
        self.input = input
        srng = T.shared_randomstreams.RandomStreams()
        mask = srng.binomial(n=1, p=numpy.float32(1-rate), size=input.shape, dtype='float32')
        self.output = T.switch(istrain, mask*self.input, self.input*numpy.float32(1-rate))
        self.params = []

    def save(self, prefix):
        pass
