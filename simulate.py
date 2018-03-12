import numpy as np

# from numpy.linalg import inv
# import math
# from numpy import linalg as LA
# import matplotlib.pyplot as plt
# import scipy
# from scipy.stats import norm


def generate_dirichlet(alpha):
    return np.random.dirichlet(alpha)

def generate_multinomial(theta):
    one_hot = np.random.multinomial(1, theta)
    return np.nonzero(one_hot)[0][0]  #where the 1 in the [0,..0,1,0,..0] is

def generate_mvn(beta_last, K, V, sigma = 0.1):
    # same as reparametrization trick
    return np.random.normal(loc = 0, scale = sigma, size = (K,V)) + beta_last

def exp_normalize(vec):
    vec_exp = np.exp(vec)
    z = sum(vec_exp)
    return vec_exp/z

def normalize_pi(beta_t): #given a beta slize (KxV) at time t, normalize it after exponenting
    return list(map(lambda vec: exp_normalize(vec), beta_t))

def simulate_data(K, V, N, D, T, sigma, alpha_0):
    theta = np.zeros(K)
    z = np.zeros(N)
    document = np.zeros((T, D, N))


    beta = np.zeros((T,K,V))
    beta_pi = np.zeros((T,K,V))
    beta[0] = generate_mvn(np.zeros((K,V)), K, V, sigma = 1) #
    beta_pi[0] = normalize_pi(beta[0])
    for t in range(1,T):
        beta[t] = generate_mvn(beta[t-1], K, V, sigma = sigma)
        beta_pi[t] = normalize_pi(beta[t])
        
    for t in range(T):
        for d in range(D):
            theta = generate_dirichlet(alpha_0) #the current document's topic mixture
            for n in range(N):
                z = generate_multinomial(theta) #topic assignment to 1 word
                document[t][d][n] = generate_multinomial(beta_pi[t][z]) #specific word assignment
    document = document.astype(int)
    print ("topics= %d, vocab = %d, D = %d, N = %d, T =%d" %(K,V,D,N,T))
    return (document, beta)