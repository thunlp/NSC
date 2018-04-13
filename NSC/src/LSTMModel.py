#-*- coding: UTF-8 -*-  
from EmbLayer import EmbLayer
from LSTMLayer import LSTMLayer
from HiddenLayer import HiddenLayer
from Update import AdaUpdates
from PoolLayer import *
from SentenceSortLayer import *
import cPickle
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

        self.doc_emb = numpy.empty(shape=[trainset.num_doc,200], dtype=numpy.float32)
        self.doc_emb_test = numpy.empty(shape=[testset.num_doc,200], dtype=numpy.float32)
        self.pred_test = numpy.empty(shape=[testset.num_doc,], dtype=numpy.float32)

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
        
        docrepresentation = layers[5].output

        pred = T.argmax(layers[-1].output, axis=1)
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
            outputs=[cost,docrepresentation],
            updates=updates,
        )

        self.test_model = theano.function(
            inputs=[docs, label,length,sentencenum,wordmask,sentencemask,maxsentencenum],
            outputs=[correct, mse, docrepresentation, pred],
        )

    def train(self, iters):
        lst = numpy.random.randint(self.trainset.epoch, size = iters)
        n = 0
        for i in lst:
            n += 1
            # print 'docs shape:'
            # print numpy.shape(self.trainset.docs[i])
            # #print 'expandlen shape:'
            # #print numpy.shape(self.trainset.expandlen[i])
            # print 'label shape:'
            # print numpy.shape(self.trainset.label[i])
            # print 'wordmask shape:'
            # print numpy.shape(self.trainset.wordmask[i])
            # print 'sentencemask shape:'
            # print numpy.shape(self.trainset.sentencemask[i])
            # print 'maxsentencenum shape:'
            # print numpy.shape(self.trainset.maxsentencenum[i])
            [out, docrepresentation] = self.train_model(self.trainset.docs[i], self.trainset.label[i], self.trainset.length[i],self.trainset.sentencenum[i],self.trainset.wordmask[i],self.trainset.sentencemask[i],self.trainset.maxsentencenum[i])
            # print n, 'cost:',out
            # print 'docrepresen, shape: ', numpy.shape(docrepresentation)
            self.doc_emb[i*16:(i+1)*16,:] = docrepresentation
            # if (n % 100 == 0):
            #     self.save_doc_emb(self.doc_emb)
        
    def test(self):
        cor = 0
        tot = 0
        mis = 0
        for i in xrange(self.testset.epoch):
            tmp = self.test_model(self.testset.docs[i], self.testset.label[i], self.testset.length[i],self.testset.sentencenum[i],self.testset.wordmask[i],self.testset.sentencemask[i],self.testset.maxsentencenum[i])
            cor += tmp[0]
            mis += tmp[1]
            tot += len(self.testset.label[i])
            self.doc_emb_test[i*16:(i+1)*16,:] = tmp[2]
            self.pred_test[i*16:(i+1)*16] = tmp[3]
        print 'Accuracy:',float(cor)/float(tot),'RMSE:',numpy.sqrt(float(mis)/float(tot))
        return cor, mis, tot


    def save(self, prefix):
        prefix += '/'
        for layer in self.layers:
            layer.save(prefix)
    
    def save_doc_emb(self, doc_emb):
        f = file('../nsc_emb_doc_train.save', 'wb')
        cPickle.dump(doc_emb, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '-> saved doc embedding training'
        
    def save_doc_emb_test(self, doc_emb):
        f = file('../nsc_emb_doc_test.save', 'wb')
        cPickle.dump(doc_emb, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '-> saved doc embedding testing'

    def save_pred_test(self, result):
        f = file('../nsc_pred_test_lstm.save', 'wb')
        cPickle.dump(result, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '--> saved final prediction: test_data'

    def load_doc_emb(self):
        f = file('../emb_doc.save', 'rb')
        result = cPickle.load(f)
        f.close()
        return result
