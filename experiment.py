import numpy as np
from functions import *
import sys
import multiprocessing as mp
from multiprocessing import Pool, get_context # Work on MacBook Air (M1, 2020, 8 cores).
import time 
import datetime
import logging
from gllim import GLLiM    

start_time = time.time() # Calculate the runtime of a programme in Python.

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
# print(ns)
# print("Chose Model " + str(model))
# print(model)

def sample_GLLiM0(n, seed = 0):
    """ Sample from the GLLiM. """
    # theta, sigma, pi, c, Gamma, A, b, nu = get_params(n)
    pi, c, Gamma, A, b, nu = get_params(n)
    
    return sample_GLLiM(pi, c, Gamma, A, b, nu, n, seed)
        

def init_params_GLLiM(n, K):
    """ Starting values for EM algorithm. """
    # theta0, sigma0, pi0, c0, Gamma0, A0, b0, nu0 = get_params(n)
    pi0, c0, Gamma0, A0, b0, nu0 = get_params(n)
    
    # theta_start = np.empty([K,d+1])
    # sigma_start = np.empty([K,d+1,d+1])
    
    pi_start    = np.empty([K])
    c_start     = np.empty([K,d])
    Gamma_start = np.empty([K,d,d])
    A_start     = np.empty([K,l,d])
    b_start     = np.empty([K,l])
    nu_start    = np.empty([K,l,l])
    
    inds = range(K0)

    # Make a partition of starting values near the true components.
    while True:
        s_inds = np.random.choice(inds, size=K)
        unique,counts = np.unique(s_inds, return_counts=True)

        if unique.size==K0:
            break
    
    for i in range(K):
        
        # theta_start[i,:]   = theta0[s_inds[i],:] + np.random.normal(0, 0.005*n**(-0.083), size=(1,d+l))
        # sigma_start[i,:,:] = sigma0[s_inds[i],:,:] + np.diag(np.abs(np.random.normal(0, 0.0005*n**(-0.25), size=d+l)))
        
        pi_start[i]        = pi0[s_inds[i]]/counts[s_inds[i]]
        c_start[i,:]       = c0[s_inds[i],:] + np.random.normal(0, 0.005*n**(-0.083), size=(1,d))
        Gamma_start[i,:,:] = Gamma0[s_inds[i],:,:] + np.diag(np.abs(np.random.normal(0, 0.0005*n**(-0.25), size=d)))
        A_start[i,:,:]     = A0[s_inds[i],:,:] + np.random.normal(0, 0.005*n**(-0.083), size=(l,d))
        b_start[i,:]       = b0[s_inds[i],:] + np.random.normal(0, 0.005*n**(-0.083), size= (1,l))
        nu_start[i,:,:]    = nu0[s_inds[i],:,:] + np.diag(np.abs(np.random.normal(0, 0.0005*n**(-0.25), size=l)))
        
    # return (theta_start, sigma_start, pi_start, c_start, Gamma_start, A_start, b_start, nu_start)
    return (pi_start, c_start, Gamma_start, A_start, b_start, nu_start)

# Main EM algorithm.   



def process_chunk_GLLiM(bound):
    """ Run EM on a range of sample sizes. """
    ind_low = bound[0]
    ind_high= bound[1]

    m = ind_high - ind_low

    seed_ctr = 2023 * ind_low   # Random seed
    
    chunk_pi    = np.empty((m, reps, K))
    chunk_c     = np.empty((m, reps, K,d))
    chunk_Gamma = np.empty((m, reps, K,d,d))
    chunk_A     = np.empty((m, reps, K,l,d))
    chunk_b     = np.empty((m, reps, K,l))
    
    if fitGLLiM == 0:
        chunk_nu = np.empty((m, reps, K,l,l))
    else:
        chunk_nu = np.empty((m, reps, K))
    chunk_iters = np.empty((m, reps))

    for i in range(ind_low, ind_high):
        n = int(ns[i])

        for rep in range(reps):
            np.random.seed(seed_ctr)
            # Sample from the mixture. 
            X, Y = sample_GLLiM0(n, seed_ctr)

            np.random.seed(seed_ctr+1)
            # theta_start, sigma_start, pi_start, c_start, Gamma_start, A_start,\
            #     b_start, nu_start = init_params_GLLiM(n,K)
            pi_start, c_start, Gamma_start, A_start,\
                b_start, nu_start = init_params_GLLiM(n,K)
            
            if fitGLLiM == 0:
                
                # Using em_GLLiM via a partition of starting values near the true components.
                out = em_GLLiM(X, Y, pi_start, c_start, Gamma_start, A_start,\
                               b_start, nu_start, max_iter=max_iter, eps=eps)
                
                logging.warning('Model ' + str(model) + ', rep:' + str(rep) +\
                                ', n:' + str(n) + ", nind:" + str(i) + ", iters:" + str(out[-2]))
                chunk_nu[i-ind_low, rep, :, :, :]    = out[5]
                
            else:
                ## Import GLLiM class.
                # print("seed_ctr = ", seed_ctr)
                gllim = GLLiM(K, l, d, n, seed_ctr)
                out = gllim.fit(X, Y, max_iter, true_init = None, random_state = seed_ctr)
                
                logging.warning('Model ' + str(model) + ', rep:' + str(rep) +\
                                ', n:' + str(n) + ", nind:" + str(i) + ", iters:" + str(out[-2]))
                chunk_nu[i-ind_low, rep, :]          = out[5]
                
            chunk_pi[i-ind_low, rep, :]          = out[0]
            chunk_c[i-ind_low, rep, :, :]        = out[1]
            chunk_Gamma[i-ind_low, rep, :, :, :] = out[2] 
            chunk_A[i-ind_low, rep, :, :, :]     = out[3]   
            chunk_b[i-ind_low, rep, :, :]        = out[4]
            chunk_iters[i-ind_low, rep]          = out[6]   

            seed_ctr += 1

    return (chunk_pi, chunk_c, chunk_Gamma, chunk_A, chunk_b, chunk_nu, chunk_iters)

# Multiprocessing.

proc_chunks_GLLiM = []

Del = n_num // n_proc 

for i in range(n_proc):
    if i == n_proc-1:
        proc_chunks_GLLiM.append(( (n_proc-1) * Del, n_num) )

    else:
        proc_chunks_GLLiM.append(( (i*Del, (i+1)*Del ) ))

if n_proc == 1: # For quick test.
    proc_chunks_GLLiM = [(99, 100)]
    
elif n_proc == 8: # 8 Cores
    proc_chunks_GLLiM = [(0, 25), (25, 40), (40, 55), (55, 70), (70, 82), (82, 90),\
                    (90, 96), (96, 100)] # 8 Cores for 100 different choices of n.

elif n_proc == 12: # 12 Cores
    proc_chunks_GLLiM = [(0, 25), (25, 40), (40, 50), (50, 60), (60, 67), (67, 75),\
                    (75, 80), (80, 85), (85, 90), (90, 94), (94, 97), (97, 100)]        
else:
    print("Please modify proc_chunks according to the core of your computer.!")
    
with get_context("fork").Pool(processes=n_proc) as pool:  # Work on MacBook Air (M1, 2020).
    proc_results_GLLiM = [pool.apply_async(process_chunk_GLLiM,
                                      args=(chunk,))
                    for chunk in proc_chunks_GLLiM]

    result_chunks_GLLiM = [r.get() for r in proc_results_GLLiM]

# Save the result for GLLiM.
done_pi    = np.concatenate([result_chunks_GLLiM[j][0] for j in range(n_proc)], axis=0)
done_c     = np.concatenate([result_chunks_GLLiM[j][1] for j in range(n_proc)], axis=0)
done_Gamma = np.concatenate([result_chunks_GLLiM[j][2] for j in range(n_proc)], axis=0)
done_A     = np.concatenate([result_chunks_GLLiM[j][3] for j in range(n_proc)], axis=0)
done_b     = np.concatenate([result_chunks_GLLiM[j][4] for j in range(n_proc)], axis=0)
done_nu    = np.concatenate([result_chunks_GLLiM[j][5] for j in range(n_proc)], axis=0)
done_iters = np.concatenate([result_chunks_GLLiM[j][6] for j in range(n_proc)], axis=0)


np.save("results_GLLiM/result_model" + str(model) +"_K" + str(K) +"_n" +\
        str(int(ns[-1])) +"_rep" + str(reps)+ "_pi.npy", done_pi)
np.save("results_GLLiM/result_model" + str(model) +"_K" + str(K) +"_n" +\
        str(int(ns[-1])) +"_rep" + str(reps)+ "_c.npy", done_c)
np.save("results_GLLiM/result_model" + str(model) +"_K" + str(K) +"_n" +\
        str(int(ns[-1])) +"_rep" + str(reps)+ "_Gamma.npy", done_Gamma)
np.save("results_GLLiM/result_model" + str(model) +"_K" + str(K) +"_n" +\
        str(int(ns[-1])) +"_rep" + str(reps)+ "_A.npy", done_A)
np.save("results_GLLiM/result_model" + str(model) +"_K" + str(K) +"_n" +\
        str(int(ns[-1])) +"_rep" + str(reps)+ "_b.npy", done_b)
np.save("results_GLLiM/result_model" + str(model) +"_K" + str(K) +"_n" +\
        str(int(ns[-1])) +"_rep" + str(reps)+ "_nu.npy", done_nu)
np.save("results_GLLiM/result_model" + str(model) +"_K" + str(K) +"_n" +\
        str(int(ns[-1])) +"_rep" + str(reps)+ "_iters.npy", done_iters)

print("--- %s seconds ---" % (time.time() - start_time))
