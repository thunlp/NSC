#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

class PrdEmbLayer(object):
    def __init__(self, rng, n_prd, dim, name, prefix=None):
        self.name = name
        if prefix == None:
            P_values = numpy.asarray(                                              
                rng.normal(scale=0.1, size=(n_prd+1, dim)),
                dtype=numpy.float32
            )
            #P_values = numpy.zeros((n_prd+1,dim),dtype=numpy.float32)
            P = theano.shared(value=P_values, name='P', borrow=True)
        else:
            f = file(prefix + name + '.save', 'rb')
            P = cPickle.load(f)
            f.close()
        self.P = P
        self.output = self.P
        self.params = [self.P]

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
