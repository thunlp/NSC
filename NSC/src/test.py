#-*- coding: UTF-8 -*-  
import sys
from Dataset import *
from LSTMModel import LSTMModel

dataname = sys.argv[1]
classes = sys.argv[2]
voc = Wordlist('../data/'+dataname+'/wordlist.txt')

testset = Dataset('../data/'+dataname+'/test.txt', voc)
trainset = []
print 'data loaded.'

model = LSTMModel(voc.size, trainset, testset, dataname, classes, '../model/'+dataname+'/bestmodel')
print 'model loaded.'
model.test()
