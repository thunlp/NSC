
#-*- coding: UTF-8 -*-  
import sys
from Dataset import *
from LSTMModel import LSTMModel

dataname = sys.argv[1]
classes = sys.argv[2]
# voc = Wordlist('../data/'+dataname+'/wordlist.txt')
# usrdict = Usrlist('../data/'+dataname+'/usrlist.txt')
# prddict = Prdlist('../data/'+dataname+'/prdlist.txt')

# trainset = Dataset('../data/'+dataname+'/train.txt', voc, usrdict, prddict)
# devset = Dataset('../data/'+dataname+'/dev.txt', voc, usrdict, prddict)
# print 'data loaded.'

print 'loading data.'
voc = Wordlist('../../../data/'+dataname+'/wordlist.txt')
usrdict = Usrlist('../../../data/'+dataname+'/usrlist.txt')
prddict = Prdlist('../../../data/'+dataname+'/prdlist.txt')

trainset = Dataset('../../../data/'+dataname+'/train.txt', voc, usrdict, prddict, maxbatch = 16)
devset = Dataset('../../../data/'+dataname+'/test.txt', voc, usrdict, prddict, maxbatch = 16)
print 'data loaded.'

model = LSTMModel(voc.size, usrdict.size, prddict.size, trainset, devset, dataname, classes, None)
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
        model.save_pred_test(model.pred_test)
        print '--> better accuracy! saved doc_emb, model and pred result on test set'
print 'bestmodel saved!'

