#-*- coding: UTF-8 -*-  
from EmbLayer import EmbLayer
from UsrEmbLayer import UsrEmbLayer
from PrdEmbLayer import PrdEmbLayer
from GetuEmbLayer import GetuEmbLayer
from GetpEmbLayer import GetpEmbLayer
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
    def __init__(self, n_voc, n_usr, n_prd, trainset, testset, dataname, classes, prefix):
        if prefix != None:
            prefix += '/'
        self.trainset = trainset
        self.testset = testset

        docs = T.imatrix()
        label = T.ivector()
        usr = T.ivector()
        prd = T.ivector()
        wordmask = T.fmatrix()
        sentencemask = T.fmatrix()
        maxsentencenum = T.iscalar()
        isTrain = T.iscalar()

        rng = numpy.random

        layers = []
        docsemb = EmbLayer(rng, docs, n_voc, 200, 'emblayer', dataname, prefix)
        Uemb = UsrEmbLayer(rng, n_usr, 200, 'usremblayer', prefix)
        Pemb = PrdEmbLayer(rng, n_prd, 200, 'prdemblayer', prefix)
        layers.append(docsemb)
        layers.append(Uemb)
        layers.append(Pemb)
        layers.append(LSTMLayer(rng, docsemb.output, wordmask, 200, 200, 'wordlstmlayer', prefix)) 
        uemb_sentence = GetuEmbLayer(usr, Uemb.output, maxsentencenum, 'uemb_sentence', prefix)
        pemb_sentence = GetpEmbLayer(prd, Pemb.output, maxsentencenum, 'pemb_sentence', prefix)
        layers.append(AttentionLayer(rng, layers[-1].output, uemb_sentence.output, pemb_sentence.output, wordmask, 200,200,200,200, 'wordattentionLayer', prefix))
        layers.append(SentenceSortLayer(layers[-1].output, maxsentencenum, prefix))
        layers.append(LSTMLayer(rng, layers[-1].output, sentencemask, 200, 200, 'sentencelstmlayer', prefix))
        uemb_doc = GetuEmbLayer(usr, Uemb.output, maxsentencenum, 'uemb_doc', prefix)
        pemb_doc = GetpEmbLayer(prd, Pemb.output, maxsentencenum, 'pemb_doc', prefix)
        layers.append(AttentionLayer(rng, layers[-1].output, uemb_doc.output, pemb_doc.output, sentencemask, 200,200,200,200, 'sentenceattentionLayer', prefix))
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
        for param in params[3:]:
            cost += T.sum(L2_rate * (param * param), acc_dtype='float32')
        gparams = [T.grad(cost, param) for param in params]

        updates = AdaUpdates(params, gparams, 0.95, 1e-6)

        self.train_model = theano.function(
            inputs=[docs, label, usr, prd, wordmask,sentencemask,maxsentencenum],
            outputs=cost,
            updates=updates,
        )

        self.test_model = theano.function(
            inputs=[docs, label, usr, prd, wordmask,sentencemask,maxsentencenum],
            outputs=[correct, mse],
        )

    def train(self, iters):
        lst = numpy.random.randint(self.trainset.epoch, size = iters)
        n = 0
        for i in lst:
            n += 1
            out = self.train_model(self.trainset.docs[i], self.trainset.label[i],self.trainset.usr[i], self.trainset.prd[i],self.trainset.wordmask[i],self.trainset.sentencemask[i],self.trainset.maxsentencenum[i])
            print n, 'cost:',out
        
    def test(self):
        cor = 0
        tot = 0
        mis = 0
        for i in xrange(self.testset.epoch):
            tmp = self.test_model(self.testset.docs[i], self.testset.label[i],self.testset.usr[i], self.testset.prd[i], self.testset.wordmask[i],self.testset.sentencemask[i],self.testset.maxsentencenum[i])
            cor += tmp[0]
            mis += tmp[1]
            tot += len(self.testset.label[i])
        print 'Accuracy:',float(cor)/float(tot),'RMSE:',numpy.sqrt(float(mis)/float(tot))
        return cor, mis, tot


    def save(self, prefix):
        prefix += '/'
        for layer in self.layers:
            layer.save(prefix)
