#-*- coding: UTF-8 -*-  
import numpy as np
import theano
import theano.tensor as T

def AdaUpdates(parameters, gradients, rho, eps):
	rho = np.float32(rho)
	eps = np.float32(eps)
	
	gradients_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=np.float32), borrow=True) for p in parameters ]
	deltas_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=np.float32), borrow=True) for p in parameters ]

	gradients_sq_new = [ rho*g_sq + (np.float32(1)-rho)*(g*g) for g_sq,g in zip(gradients_sq,gradients) ]
	deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in zip(deltas_sq,gradients_sq_new,gradients) ]

	deltas_sq_new = [ rho*d_sq + (np.float32(1)-rho)*(d*d) for d_sq,d in zip(deltas_sq,deltas) ]

	gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
	deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
	parameters_updates = [ (p,p - d) for p,d in zip(parameters,deltas) ]
	return gradient_sq_updates + deltas_sq_updates + parameters_updates
