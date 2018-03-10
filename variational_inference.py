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


class variational_inference():
	# variational papareters are:
	# q(beta) ~ N(beta|mu_beta, cov_beta)  for each (t, k)
	# q(z) ~ Mult(phi) for each (t,d,n). phi is k-vector with sum = 1
	# q(theta) ~ Dirichlet(alpha) for each (d, t). alpha is k-vector
	def __init__(self):
		self.