import sys
import numpy as np
from discrepancies import *
from scipy.spatial.distance import cdist
import logging


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=1, type=int, help='Type number.')
parser.add_argument('-K', '--K', default=4, type=int, help='Number of mixture components.')
parser.add_argument('-r' ,'--reps', default=20, type=int, help='Number of replications per sample size.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.') # Work on MacBook Air (M1, 2020, 8 cores)
parser.add_argument('-mi','--maxit', default=2000, type=int, help='Maximum EM iterations.')
parser.add_argument('-e', '--eps', default=1e-5, type=float, help='EM stopping criterion.')
parser.add_argument('-f', '--fitGLLiM', default=0, type=int, help='Fitting method for estimating GLLiM.')
parser.add_argument('-nnum', '--n_num', default=100, type=int, help='Number of different choices of sample size.')
parser.add_argument('-nsmax', '--ns_max', default=100000, type=int, help='Number of sample size maximum.')

args = parser.parse_args()

print(args)

model = args.model                    # Type number
K = args.K                            # Number of mixture components
n_proc = args.nproc                   # Number of cores to use
reps = args.reps                      # Number of replications to run per sample size
max_iter = args.maxit                 # Maximum EM iterations
eps = args.eps                        # EM Stopping criterion.
fitGLLiM = args.fitGLLiM              # Fitting method for estimating GLLiM.
n_num = args.n_num                    # Number of different choices of sample size.
ns_max = args.ns_max                    # Number of sample size maximum.

exec(open("models.py").read())

logging.basicConfig(filename='std_mod' + str(model) + '_K' + str(K) +\
                    '.log', filemode='w', format='%(asctime)s %(message)s')

# Test for 100 different choices of n between 10^2 and 10^5 on MacBook Air (M1, 2020, 8 cores). 
ns = np.concatenate([np.linspace(100, 500, 5), np.linspace(500, ns_max, n_num-5)])
    
dists = np.empty((n_num, reps))

pis    = np.load("results_GLLiM/result_model" + str(model) +"_K" + str(K) +\
                 "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_pi.npy")
cs     = np.load("results_GLLiM/result_model" + str(model) +"_K" + str(K) +\
                 "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_c.npy")
Gammas = np.load("results_GLLiM/result_model" + str(model) +"_K" + str(K) +\
                 "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_Gamma.npy")
As     = np.load("results_GLLiM/result_model" + str(model) +"_K" + str(K) +\
                 "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_A.npy")
bs     = np.load("results_GLLiM/result_model" + str(model) +"_K" + str(K) +\
                 "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_b.npy")
nus    = np.load("results_GLLiM/result_model" + str(model) +"_K" + str(K) +\
                 "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_nu.npy")

for i in range(n_num):
    pi0, c0, Gamma0, A0, b0, nu0 = get_params(ns[i])
    
    for j in range(reps):
        if fitGLLiM == 0:
            if ((model == 1) or (model == 3)):
                dists[i,j] = gauss_loss_GLLiM1(pi0, c0, Gamma0, A0, b0, nu0, pis[i,j,:],\
                                        cs[i,j,:,:], Gammas[i,j,:,:,:], As[i,j,:,:,:],\
                                            bs[i,j,:,:], nus[i,j,:,:,:])
            elif ((model == 2) or (model == 4)):
                dists[i,j] = gauss_loss_GLLiM2(pi0, c0, Gamma0, A0, b0, nu0, pis[i,j,:],\
                                        cs[i,j,:,:], Gammas[i,j,:,:,:], As[i,j,:,:,:],\
                                            bs[i,j,:,:], nus[i,j,:,:,:])         
            else:
                sys.exit("Model unrecognized.")        
        else:    
            dists[i,j] = gauss_loss_GLLiM1(pi0, c0, Gamma0, A0, b0, nu0, pis[i,j,:],\
                                    cs[i,j,:,:], Gammas[i,j,:,:,:], As[i,j,:,:,:],\
                                        bs[i,j,:,:], nus[i,j,:].reshape(K, l, l))

np.save("results_GLLiM/result_model" + str(model) +"_K" + str(K) +"_n" +\
        str(int(ns[-1])) +"_rep" + str(reps)+ "_loss.npy", dists)
