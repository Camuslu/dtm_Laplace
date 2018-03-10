import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import scipy
from scipy.stats import wishart
from scipy.stats import bernoulli
from scipy.stats import norm
from scipy import misc
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der


class variational_inference():
	# variational papareters are:
	# q(beta) ~ N(beta|mu_beta, cov_beta)  for each (t, k)
	# q(z) ~ Mult(phi) for each (t,d,n). phi is k-vector with sum = 1
	# q(theta) ~ Dirichlet(alpha) for each (d, t). alpha is k-vector
	def __init__(self, T, D, K, iters, document):
		self.alpha=np.ones((T,D,K)) #the uniform prior on Dirichlet, to be updated later
		self.mu_beta_t = np.zeros((T, K, V))
		self.cov_beta_t = np.zeros((T, K, V, V))
		self.phi = np.zeros((T, D, N, K))
		self.E_log_sum_exp_beta_t_k=np.zeros((T,K))
		self.iteration = iters
		self.document = document
		self.token_word = {}
		self.word_token = {}

	# TODO: tokenize a document into {"apple": 1,...}
	def tokenize(self, document):
		pass


	def s_func(self, beta_k):
	    return np.exp(beta_k)

	def t_func(self, beta_k):
		return sum(np.exp(self.beta_k[v]))

	def B_func(self, k, t):
	    B = np.zeros((V))
	    for d in range(D):
	        for n in range(N):
	            B[int(self.document[t][d][n])] += self.phi[t][d][n][k]
    	return B

    def gradient_descent_beta_0(self, k):
    	# find mode of f(beta_0) for topic k
	    b_val=self.B_func(phi, k, 0)
	    fn = lambda beta: sum(b_val)*misc.logsumexp(beta) + 0.5*np.dot(beta,beta)+0.5/(sigma ** 2)*np.dot(beta,beta) \
	    				  -1.0 / (sigma ** 2)*np.dot(mu_beta_t[1][k],beta)-np.dot(b_val,beta)
	    res = minimize(fn, np.zeros(V), method='SLSQP')
	    return res.x

	def gradient_descent_beta_t(self, t, k):
	    b_val=self.B_func(phi, k, t)
	    fn = lambda beta: sum(b_val)*misc.logsumexp(beta) + 2/(sigma**2)*np.dot(beta,beta) \
	    				  -1.0 / (sigma ** 2) * np.dot(mu_beta_t[t + 1][k]+mu_beta_t[t - 1][k], beta)-np.dot(b_val,beta)
	    res = minimize(fn, np.zeros(V), method='SLSQP')
	    return res.x

	def gradient_descent_beta_T(self, k):
		b_val = self.B_func(phi, k, T-1)
	    fn = lambda beta: sum(b_val)*misc.logsumexp(beta) + 0.5/(sigma ** 2)*np.dot(beta,beta)\
	    				  -1.0 / (sigma ** 2)*np.dot(mu_beta_t[T - 2][k],beta)-np.dot(b_val,beta)
	    res = minimize(fn, np.zeros(V), method='SLSQP')
	    return res.x

	def update_beta_0(self):
		for k in range(K):
			self.mu_beta_t[0][k] = gradient_descent_beta_0(k)
	        s = np.exp(self.mu_beta_t[0][k])
	        t = sum(s)
	        b_val = B_func(phi, k, 0)
	        ## brute-force, slow!
	        #delta_sqr = np.identity(V) + 1.0 / (sigma ** 2) * np.identity(V) \
	        #			 + (sum(b_val)/(t_func_val ** 2)) * (t * np.diag(s)- np.outer(s, s))
	        # cov_beta_t[0][k] = np.linalg.inv(delta_sqr) 
	        # 
	        D = (1+ 1/(sigma**2))*np.ones(V) + sum(b_val)*s/t * np.ones(V)
	        D_inv = 1/D_inv #array of length V, to be used as diagonal later
	        v = (np.sqrt(sum(b_val))/t)*s
	        downstair=1-sum(D_inv*v*v)
	        D_inv_v=D_inv*v
	        upstair=np.outer(D_inv_v,D_inv_v)
	        self.cov_beta_t[0][k]=np.diag(D_inv)+upstair/downstair


	        

