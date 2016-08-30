#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

class GetpEmbLayer(object):
    def __init__(self, p, Pemb, maxsentencesum, name, prefix=None):
        self.input = p
        self.name = name

        if self.name == 'pemb_sentence':
            palloc = T.alloc(p,maxsentencesum,T.shape(p)[0])
            pflatten = palloc.T.flatten()   
        else:
            pflatten = p
        
        self.output = Pemb[pflatten]
        self.params = []

    def save(self, prefix):
        pass
