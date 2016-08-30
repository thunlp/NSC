#-*- coding: UTF-8 -*-  
from EmbLayer import EmbLayer
from LSTMLayer import LSTMLayer
from HiddenLayer import HiddenLayer
from Update import AdaUpdates
from PoolLayer import *
from SentenceSortLayer import *
import theano
import theano.tensor as T
import numpy
import random
import sys

class LSTMModel(object):
    def __init__(self, n_voc, trainset, testset,dataname, classes, prefix):
        if prefix != None:
            prefix += '/'
        self.trainset = trainset
        self.testset = testset

        docs = T.imatrix()
        label = T.ivector()
        length = T.fvector()
        sentencenum = T.fvector()
        wordmask = T.fmatrix()
        sentencemask = T.fmatrix()
        maxsentencenum = T.iscalar()
        isTrain = T.iscalar()

        rng = numpy.random

        layers = []
        layers.append(EmbLayer(rng, docs, n_voc, 200, 'emblayer', dataname, prefix))
        layers.append(LSTMLayer(rng, layers[-1].output, wordmask, 200, 200, 'wordlstmlayer', prefix)) 
        layers.append(MeanPoolLayer(layers[-1].output, length))
        layers.append(SentenceSortLayer(layers[-1].output,maxsentencenum))
        layers.append(LSTMLayer(rng, layers[-1].output, sentencemask, 200, 200, 'sentencelstmlayer', prefix))
        layers.append(MeanPoolLayer(layers[-1].output, sentencenum))
        layers.append(HiddenLayer(rng, layers[-1].output, 200, 200, 'fulllayer', prefix))
        layers.append(HiddenLayer(rng, layers[-1].output, 200, int(classes), 'softmaxlayer', prefix, activation=T.nnet.softmax))
        self.layers = layers
        
        cost = -T.mean(T.log(layers[-1].output)[T.arange(label.shape[0]), label], acc_dtype='float32')
        correct = T.sum(T.eq(T.argmax(layers[-1].output, axis=1), label), acc_dtype='int32')
        err = T.argmax(layers[-1].output, axis=1) - label
        mse = T.sum(err * err)
        
        params = []
        for layer in layers:
            params += layer.params
        L2_rate = numpy.float32(1e-5)
        for param in params[1:]:
            cost += T.sum(L2_rate * (param * param), acc_dtype='float32')
        gparams = [T.grad(cost, param) for param in params]

        updates = AdaUpdates(params, gparams, 0.95, 1e-6)

        self.train_model = theano.function(
            inputs=[docs, label,length,sentencenum,wordmask,sentencemask,maxsentencenum],
            outputs=cost,
            updates=updates,
        )

        self.test_model = theano.function(
            inputs=[docs, label,length,sentencenum,wordmask,sentencemask,maxsentencenum],
            outputs=[correct, mse],
        )

    def train(self, iters):
        lst = numpy.random.randint(self.trainset.epoch, size = iters)
        n = 0
        for i in lst:
            n += 1
            out = self.train_model(self.trainset.docs[i], self.trainset.label[i], self.trainset.length[i],self.trainset.sentencenum[i],self.trainset.wordmask[i],self.trainset.sentencemask[i],self.trainset.maxsentencenum[i])
            print n, 'cost:',out
        
    def test(self):
        cor = 0
        tot = 0
        mis = 0
        for i in xrange(self.testset.epoch):
            tmp = self.test_model(self.testset.docs[i], self.testset.label[i], self.testset.length[i],self.testset.sentencenum[i],self.testset.wordmask[i],self.testset.sentencemask[i],self.testset.maxsentencenum[i])
            cor += tmp[0]
            mis += tmp[1]
            tot += len(self.testset.label[i])
        print 'Accuracy:',float(cor)/float(tot),'RMSE:',numpy.sqrt(float(mis)/float(tot))
        return cor, mis, tot


    def save(self, prefix):
        prefix += '/'
        for layer in self.layers:
            layer.save(prefix)
