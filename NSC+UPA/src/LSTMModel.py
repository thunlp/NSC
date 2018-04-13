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

        self.doc_emb = numpy.empty(shape=[trainset.num_doc,200], dtype=numpy.float32)
        self.doc_emb_test = numpy.empty(shape=[testset.num_doc,200], dtype=numpy.float32)
        self.pred_test = numpy.empty(shape=[testset.num_doc,], dtype=numpy.float32)

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

        docrepresentation = layers[7].output
        
        pred = T.argmax(layers[-1].output, axis=1)
        cost = -T.mean(T.log(layers[-1].output)[T.arange(label.shape[0]), label], acc_dtype='float32')
        correct = T.sum(T.eq(T.argmax(layers[-1].output, axis=1), label), acc_dtype='int32')
        err = T.argmax(layers[-1].output, axis=1) - label
        mse = T.sum(err * err)
        
        params = []
        for layer in layers:
            params += layer.params
        L2_rate = numpy.float32(1e-6)
        for param in params[3:]:
            cost += T.sum(L2_rate * (param * param), acc_dtype='float32')
        gparams = [T.grad(cost, param) for param in params]

        updates = AdaUpdates(params, gparams, 0.90, 1e-7)

        self.train_model = theano.function(
            inputs=[docs, label, usr, prd, wordmask,sentencemask,maxsentencenum],
            outputs=[cost, docrepresentation],
            updates=updates,
        )

        self.test_model = theano.function(
            inputs=[docs, label, usr, prd, wordmask,sentencemask,maxsentencenum],
            outputs=[correct, mse, docrepresentation, pred],
        )

    def train(self, iters):
        lst = numpy.random.randint(self.trainset.epoch, size = iters)
        n = 0
        for i in lst:
            n += 1
            [out, docrepresentation] = self.train_model(self.trainset.docs[i], self.trainset.label[i],self.trainset.usr[i], self.trainset.prd[i],self.trainset.wordmask[i],self.trainset.sentencemask[i],self.trainset.maxsentencenum[i])
            self.doc_emb[i*16:(i+1)*16,:] = docrepresentation
            # print n, 'cost:',out
        # self.save_doc_emb(self.doc_emb)
        
    def test(self):
        cor = 0
        tot = 0
        mis = 0
        for i in xrange(self.testset.epoch):
            tmp = self.test_model(self.testset.docs[i], self.testset.label[i],self.testset.usr[i], self.testset.prd[i], self.testset.wordmask[i],self.testset.sentencemask[i],self.testset.maxsentencenum[i])
            cor += tmp[0]
            mis += tmp[1]
            tot += len(self.testset.label[i])
            self.doc_emb_test[i*16:(i+1)*16,:] = tmp[2]
            self.pred_test[i*16:(i+1)*16] = tmp[3]
        print 'Accuracy:',float(cor)/float(tot),'RMSE:',numpy.sqrt(float(mis)/float(tot))
        # self.save_doc_emb_test(self.doc_emb_test)
        return cor, mis, tot


    def save(self, prefix):
        prefix += '/'
        for layer in self.layers:
            layer.save(prefix)

    def save_doc_emb(self, doc_emb):
        f = file('../nscupa_emb_doc_train.save', 'wb')
        cPickle.dump(doc_emb, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '-> saved doc embedding training'

    def save_pred_test(self, result):
        f = file('../pred_test_lstm.save', 'wb')
        cPickle.dump(result, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '--> saved final prediction: test_data'
        
    def save_doc_emb_test(self, doc_emb):
        f = file('../nscupa_emb_doc_test.save', 'wb')
        cPickle.dump(doc_emb, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '-> saved doc embedding testing'

    def load_doc_emb(self):
        f = file('../emb_doc.save', 'rb')
        result = cPickle.load(f)
        f.close()
        return result
