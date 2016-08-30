#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

def softmask(x,mask):
    y = T.exp(x)
    y =y *mask
    sumx = T.sum(y,axis=1)
    x = y/sumx.dimshuffle(0,'x')
    return x

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


class SimpleAttentionLayer(object):
    def __init__(self, rng, input,mask, n_in, n_out, name, prefix=None):
        self.input = input

        if prefix is None:
            W_values = numpy.asarray(                                              
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=numpy.float32
            )
            W = theano.shared(value=W_values, name='W', borrow=True)
            
            v_values = numpy.asarray(
                rng.normal(scale=0.1, size=(n_out,)),
                dtype=numpy.float32
            )
            v = theano.shared(value=v_values, name='v', borrow=True)
            
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)            
            b = theano.shared(value=b_values, name='b', borrow=True)

        else:
            f = file(prefix + name + '.save', 'rb')
            W = cPickle.load(f)
            v = cPickle.load(f)
            b = cPickle.load(f)
            f.close()

        self.W = W
        self.v = v
        self.b = b

        atten = T.tanh(T.dot(input, self.W)+ b)                        
        atten = T.sum(atten * v, axis=2, acc_dtype='float32')                   
        atten = softmask(atten.dimshuffle(1,0),mask.dimshuffle(1,0)).dimshuffle(1, 0)         
        output = atten.dimshuffle(0, 1, 'x') * input
        self.output = T.sum(output, axis=0, acc_dtype='float32')                
        
        self.params = [self.W,self.v,self.b]
        self.name=name
        self.atten = atten

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


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
