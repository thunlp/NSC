#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy

class SentenceSortLayer(object):
    def __init__(self, input,maxsentencenum):
        self.input = input
        [sentencelen,emblen] = T.shape(input)
        output = input.reshape((sentencelen / maxsentencenum,maxsentencenum,emblen))
        output = output.dimshuffle(1,0,2)
        self.output = output
        self.params = []
        

    def save(self, prefix):
        pass
