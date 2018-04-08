
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
# model = LSTMModel(voc.size, trainset, devset, dataname, classes, '../model/' + dataname + '/bestmodel')

model.train(5)
print '****************************************************************************'
print 'test 1'
currentresult = model.test()
print '****************************************************************************'
print '\n'
for i in xrange(1,400):
	model.train(5)
	print '****************************************************************************'
	print 'test',i+1
	newresult=model.test()
	print '****************************************************************************'
	print '\n'
	print newresult[0]
	print currentresult[0]
	if newresult[0]>currentresult[0]:
		print 'a!'
		currentresult=newresult
		# save document representation for dataset
		model.save('../model/'+dataname+'/bestmodel')
		model.save_doc_emb(model.doc_emb)
		model.save_doc_emb_test(model.doc_emb_test)
		model.save_pred_test(model.pred_test)
		print '--> better accuracy! saved doc_emb, model and pred result on test set'
print 'bestmodel saved!'

