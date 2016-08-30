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

class AttentionLayer(object):
    def __init__(self, rng, input, input_u, input_p, mask, n_wordin, n_usrin, n_prdin, n_out, name, prefix=None):
        self.input = input
        self.inputu = input_u
        self.inputp = input_p

        if prefix is None:
            W_values = numpy.asarray(                                              
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_wordin + n_out)),
                    high=numpy.sqrt(6. / (n_wordin + n_out)),
                    size=(n_wordin, n_out)
                ),
                dtype=numpy.float32
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

            '''
            v_values = numpy.zeros((n_out,), dtype=theano.config.floatX)            
            v = theano.shared(value=v_values, name='v', borrow=True)
            '''
            v_values = numpy.asarray(
                rng.normal(scale=0.1, size=(n_out,)),
                dtype=numpy.float32
            )
            v = theano.shared(value=v_values, name='v', borrow=True)
            
            Wu_values = numpy.asarray(                                              
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_usrin + n_out)),
                    high=numpy.sqrt(6. / (n_usrin + n_out)),
                    size=(n_usrin, n_out)
                ),
                dtype=numpy.float32
            )
            Wu = theano.shared(value=Wu_values, name='Wu', borrow=True)
            
            Wp_values = numpy.asarray(                                              
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_prdin + n_out)),
                    high=numpy.sqrt(6. / (n_prdin + n_out)),
                    size=(n_prdin, n_out)
                ),
                dtype=numpy.float32
            )
            Wp = theano.shared(value=Wp_values, name='Wp', borrow=True)
            
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
 
        else:
            f = file(prefix + name + '.save', 'rb')
            W = cPickle.load(f)
            v = cPickle.load(f)
            Wu = cPickle.load(f)
            Wp = cPickle.load(f)
            b = cPickle.load(f)
            f.close()

        self.W = W
        self.v = v
        self.Wu = Wu
        self.Wp = Wp
        self.b = b

        attenu = T.dot(input_u, self.Wu)                                              
        attenp = T.dot(input_p, self.Wp)                                              

        atten = T.tanh(T.dot(input, self.W)+ attenu + attenp +b)                         
        atten = T.sum(atten * v, axis=2, acc_dtype='float32')                 
        atten = softmask(atten.dimshuffle(1,0), mask.dimshuffle(1,0)).dimshuffle(1, 0)        
        output = atten.dimshuffle(0, 1, 'x') * input
        self.output = T.sum(output, axis=0, acc_dtype='float32')                

        self.params = [self.W, self.v,self.Wu,self.Wp,self.b]
        self.name=name
        self.atten = atten
        self.mask = mask


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
