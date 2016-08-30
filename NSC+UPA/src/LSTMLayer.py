#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

def randMatrix(rng, shape, lim):
    return numpy.asarray(
        rng.uniform(
            low=-lim,
            high=lim,
            size=shape
        ),
        dtype=numpy.float32
    )

class LSTMLayer(object):
    def __init__(self, rng, input, mask, n_in, n_out, name, prefix=None):
        self.input = input
        self.name = name

        limV = numpy.sqrt(6. / (n_in + n_out * 2))
        limG = limV * 4

        if prefix is None:
            Wi1_values = randMatrix(rng, (n_in, n_out), limG)
            Wi1 = theano.shared(value=Wi1_values, name='Wi1', borrow=True)
            Wi2_values = randMatrix(rng, (n_out, n_out), limG)
            Wi2 = theano.shared(value=Wi2_values, name='Wi2', borrow=True)
            bi_values = numpy.zeros((n_out,), dtype=numpy.float32)
            bi = theano.shared(value=bi_values, name='bi', borrow=True)

            Wo1_values = randMatrix(rng, (n_in, n_out), limG)
            Wo1 = theano.shared(value=Wo1_values, name='Wo1', borrow=True)
            Wo2_values = randMatrix(rng, (n_out, n_out), limG)
            Wo2 = theano.shared(value=Wo2_values, name='Wo2', borrow=True)
            bo_values = numpy.zeros((n_out,), dtype=numpy.float32)
            bo = theano.shared(value=bo_values, name='bo', borrow=True)

            Wf1_values = randMatrix(rng, (n_in, n_out), limG)
            Wf1 = theano.shared(value=Wf1_values, name='Wf1', borrow=True)
            Wf2_values = randMatrix(rng, (n_out, n_out), limG)
            Wf2 = theano.shared(value=Wf2_values, name='Wf2', borrow=True)
            bf_values = numpy.zeros((n_out,), dtype=numpy.float32)
            bf = theano.shared(value=bf_values, name='bf', borrow=True)

            Wc1_values = randMatrix(rng, (n_in, n_out), limV)
            Wc1 = theano.shared(value=Wc1_values, name='Wc1', borrow=True)
            Wc2_values = randMatrix(rng, (n_out, n_out), limV)
            Wc2 = theano.shared(value=Wc2_values, name='Wc2', borrow=True)
            bc_values = numpy.zeros((n_out,), dtype=numpy.float32)
            bc = theano.shared(value=bc_values, name='bc', borrow=True)

        else:
            f = file(prefix + name + '.save', 'rb')
            Wi1 = cPickle.load(f)
            Wi2 = cPickle.load(f)
            bi = cPickle.load(f)

            Wo1 = cPickle.load(f)
            Wo2 = cPickle.load(f)
            bo = cPickle.load(f)

            Wf1 = cPickle.load(f)
            Wf2 = cPickle.load(f)
            bf = cPickle.load(f)

            Wc1 = cPickle.load(f)
            Wc2 = cPickle.load(f)
            bc = cPickle.load(f)

            f.close()

        self.Wi1 = Wi1
        self.Wi2 = Wi2
        self.bi = bi

        self.Wo1 = Wo1
        self.Wo2 = Wo2
        self.bo = bo

        self.Wf1 = Wf1
        self.Wf2 = Wf2
        self.bf = bf

        self.Wc1 = Wc1
        self.Wc2 = Wc2
        self.bc = bc

        def step(emb, mask, C, prev):
            Gi = T.nnet.sigmoid(T.dot(emb, self.Wi1) + T.dot(prev, self.Wi2) + self.bi)
            Go = T.nnet.sigmoid(T.dot(emb, self.Wo1) + T.dot(prev, self.Wo2) + self.bo)
            Gf = T.nnet.sigmoid(T.dot(emb, self.Wf1) + T.dot(prev, self.Wf2) + self.bf)
            Ct = T.tanh(T.dot(emb, self.Wc1) + T.dot(prev, self.Wc2) + self.bc)

            CC = C * Gf + Ct * Gi
            CC = CC * mask.dimshuffle(0,'x') 
            CC = T.cast(CC,'float32')
            h = T.tanh(CC) * Go
            h = h * mask.dimshuffle(0,'x') 
            h = T.cast(h,'float32')
            return [CC, h]

        outs, _ = theano.scan(fn=step,
            outputs_info=[T.zeros_like(T.dot(input[0], self.Wi1)), T.zeros_like(T.dot(input[0], self.Wi1))],
            sequences=[input, mask])

        self.output = outs[1]

        self.params = [self.Wi1, self.Wi2, self.bi, self.Wo1, self.Wo2, self.bo,
            self.Wf1, self.Wf2, self.bf, self.Wc1, self.Wc2, self.bc]

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
