
#-*- coding: UTF-8 -*-  
import sys
from Dataset import *
from LSTMModel import LSTMModel

dataname = sys.argv[1]
classes = sys.argv[2]
print 'loading data.'
voc = Wordlist('../../../data/'+dataname+'/wordlist.txt')

trainset = Dataset('../../../data/'+dataname+'/train.txt', voc, maxbatch = 16)
devset = Dataset('../../../data/'+dataname+'/test.txt', voc, maxbatch = 16)
print 'data loaded.'

model = LSTMModel(voc.size,trainset, devset, dataname, classes, None)
model.train(100)
print '****************************************************************************'
print 'test 1'
result = model.test()
print '****************************************************************************'
print '\n'
for i in xrange(1,400):
	model.train(100)
	print '****************************************************************************'
	print 'test',i+1
	newresult=model.test()
	print '****************************************************************************'
	print '\n'
	if newresult[0]>result[0] :
		result=newresult
		model.save('../model/'+dataname+'/bestmodel')
        # save document representation for dataset
        model.save_doc_emb(model.doc_emb)
        model.save_doc_emb_test(model.doc_emb_test)
        print 'better accuracy! saved doc_emb and model'
print 'bestmodel saved!'

