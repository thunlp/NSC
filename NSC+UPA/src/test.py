#-*- coding: UTF-8 -*-  
import sys
from Dataset import *
from LSTMModel import LSTMModel

dataname = sys.argv[1]
classes = sys.argv[2]
voc = Wordlist('../data/'+dataname+'/wordlist.txt')
usrdict = Usrlist('../data/'+dataname+'/usrlist.txt')
prddict = Prdlist('../data/'+dataname+'/prdlist.txt')

testset = Dataset('../data/'+dataname+'/test.txt', voc, usrdict, prddict)
trainset = []
print 'data loaded.'

model = LSTMModel(voc.size, usrdict.size, prddict.size, trainset, testset, dataname, classes, '../model/'+dataname+'/bestmodel')
print 'model loaded.'
model.test()
