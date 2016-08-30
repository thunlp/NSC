#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

class GetuEmbLayer(object):
    def __init__(self, u, Uemb, maxsentencesum, name, prefix=None):
        self.input = u
        self.name = name

        if self.name == 'uemb_sentence':
            ualloc = T.alloc(u,maxsentencesum,T.shape(u)[0])
            uflatten = ualloc.T.flatten()   
        else:
            uflatten = u
        
        self.output = Uemb[uflatten]
        self.params = []

    def save(self, prefix):
        pass
