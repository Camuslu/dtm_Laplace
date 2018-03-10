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
	def __init__(self, T, D, K, N, iters, document,alpha_0 = None, sample_size = 1000):
		self.alpha=np.ones((T,D,K)) #the uniform prior on Dirichlet, to be updated later
		self.mu_beta_t = np.zeros((T, K, V))
		self.cov_beta_t = np.zeros((T, K, V, V))
		self.phi = np.zeros((T, D, N, K))
		self.T = T
		self.K = K
		self.D = D
		self.N = N
		self.E_log_sum_exp_beta_t_k=np.zeros((T,K))
		self.iteration = iters
		self.document = document
		self.token_word = {}
		self.word_token = {}
		if alpha_0 == None:
			self.alpha_0 = np.ones(self.K) #the uniform prior on Dirichlet
		self.S = sample_size

	# TODO: tokenize a document into {"apple": 1,...}
	def tokenize(self, document):
		pass


	def s_func(self, beta_k):
	    return np.exp(beta_k)

	def t_func(self, beta_k):
		return sum(np.exp(self.beta_k[v]))

	def B_func(self, k, t):
	    B = np.zeros((self.V))
	    for d in range(self.D):
	        for n in range(self.N):
	            B[int(self.document[t][d][n])] += self.phi[t][d][n][k]
    	return B

    def gradient_descent_beta_0(self, k):
    	# find mode of f(beta) for topic k, or argmin of -f(beta)
	    b_val=self.B_func(phi, k, 0)
	    fn = lambda beta: sum(b_val)*misc.logsumexp(beta) + 0.5*np.dot(beta,beta)+0.5/(self.sigma ** 2)*np.dot(beta,beta) \
	    				  -1.0 / (self.sigma ** 2)*np.dot(mu_beta_t[1][k],beta)-np.dot(b_val,beta)
	    res = minimize(fn, np.zeros(V), method='SLSQP')
	    return res.x

	def gradient_descent_beta_t(self, t, k):
	    b_val=self.B_func(phi, k, t)
	    fn = lambda beta: sum(b_val)*misc.logsumexp(beta) + 2/(sigma**2)*np.dot(beta,beta) \
	    				  -1.0 / (self.sigma ** 2) * np.dot(mu_beta_t[t + 1][k]+mu_beta_t[t - 1][k], beta)-np.dot(b_val,beta)
	    res = minimize(fn, np.zeros(V), method='SLSQP')
	    return res.x

	def gradient_descent_beta_T(self, k):
		b_val = self.B_func(phi, k, T-1)
	    fn = lambda beta: sum(b_val)*misc.logsumexp(beta) + 0.5/(self.sigma ** 2)*np.dot(beta,beta)\
	    				  -1.0 / (self.sigma ** 2)*np.dot(mu_beta_t[T - 2][k],beta)-np.dot(b_val,beta)
	    res = minimize(fn, np.zeros(V), method='SLSQP')
	    return res.x

	def update_beta_0(self):
		for k in range(self.K):
			self.mu_beta_t[0][k] = self.gradient_descent_beta_0(k)
	        s = np.exp(self.mu_beta_t[0][k])
	        t = sum(s)
	        b_val = self.B_func(phi, k, 0)
	        D = (1+ 1/(self.sigma**2))*np.ones(V) + sum(b_val)*s/t 
	        D_inv = 1/D_inv #array of length V, to be used as diagonal later
	        v = (np.sqrt(sum(b_val))/t)*s
	        downstair=1-sum(D_inv*v*v)
	        D_inv_v=D_inv*v
	        upstair=np.outer(D_inv_v,D_inv_v)
	        self.cov_beta_t[0][k]=np.diag(D_inv)+upstair/downstair

	def update_beta_t(self):
		for t in range(1, self.T - 1):
	        for k in range(self.K):
	            self.mu_beta_t[t][k] = self.gradient_descent_beta_t(k, t)
	            s = np.exp(self.mu_beta_t[t][k])
		        t = sum(s)
		        b_val = self.B_func(phi, k, t)
	         	D = 2/(self.sigma**2)*np.ones(V) + sum(b_val)*s/t
		        D_inv = 1/D_inv #array of length V, to be used as diagonal later
		        v = (np.sqrt(sum(b_val))/t)*s
		        downstair=1-sum(D_inv*v*v)
		        D_inv_v=D_inv*v
		        upstair=np.outer(D_inv_v,D_inv_v)
		        self.cov_beta_t[t][k]=np.diag(D_inv)+upstair/downstair
	
	def update_beta_T(self):
		for k in range(self.K):
			self.mu_beta_t[0][k] = self.gradient_descent_beta_T(k)
	        s = np.exp(self.mu_beta_t[self.T-1][k])
	        t = sum(s)

	        D = 1/(self.sigma**2)*np.ones(V) + sum(b_val)*s/t
	        D_inv = 1/D_inv #array of length V, to be used as diagonal later
	        v = (np.sqrt(sum(b_val))/t)*s
	        downstair=1-sum(D_inv*v*v)
	        D_inv_v=D_inv*v
	        upstair=np.outer(D_inv_v,D_inv_v)
	        self.cov_beta_t[T-1][k]=np.diag(D_inv)+upstair/downstair   


	def update_alpha(self):
	    for t in range(self.T):
	        for d in range(self.D):
	            self.alpha[t][d] = self.alpha_0+np.sum(self.phi[t][d],axis=0)
	            #no need to normalize the alpha k-vector here. it's just a parameter for Dirichlet

	def update_phi(self):
		for t in range(self.T):
	        for d in range(self.D):
	            quick_digamma = sp.digamma(sum(self.alpha[t][d]))  #this is the same for every word n on every topic k
	            for n in range(self.N):
	                phi_temp = np.zeros(self.K)
	                log_phi_temp = np.zeros(self.K)
	                word_index = int(document[t][d][n])
	                for k in range(self.K):
	                    p1 = sp.digamma(self.alpha[t][d][k])-quick_digamma  #log of the contribution from theta
	                    p2 = self.mu_beta_t[t][k][word_index]-self.E_log_sum_exp_beta_t_k[t][k] #log of the contribution from beta
	                    log_phi_temp[k] = p1+p2
	                #in case of overflow, every term in the log space minus the max log, then exponent. Finally normalize the vector phi_temp
	                #Since it's ‘proportional to’, this transformation preserves the proportion amongst all terms without causing overflow
	                max_log = max(log_phi_temp)
	                log_phi_temp = log_phi_temp-max_log
	                phi_temp = np.exp(log_phi_temp)
	                self.phi[t][d][n] = phi_temp/sum(phi_temp) #normalize

	def update_EXP_log_sum_exp_beta_t_k(self):
		for t in range(self.T):
	        for k in range(self.K):
	            samples = [self.quick_sample_Gaussian(t,k) for _ in range(S)]
	            self.E_log_sum_exp_beta_t_k[t][k] = sum(samples_sum)/self.S

	def quick_sample_Gaussian(self,t,k):
		# first draw [x1...xV] from N(0,1) 
		# then multiply each xi with sigma_i, where sigma_i is sqrt of covariance at beta_t_k (here we use the diagonal approx for covariance matrix)
		# finally add mu_beta_t_k for translation
		x = np.random.normal(0, 1, self.V)
		sigma_ = np.sqrt(np.diag(self.cov_beta_t[t][k]))
		beta_s  = lambda_*x + self.mu_beta_t[t][k]
		return misc.logsumexp(x)