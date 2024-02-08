import numpy as np
from functions import *
import matplotlib.pyplot as plt
import matplotlib
import logging

print("Log-log scale plots for the simulation results of CRPE-GLLiM")

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

text_size = 17

reps = 20
lw = 2.5
elw = 0.8

matplotlib.rc('text', usetex=True)
matplotlib.rc('xtick', labelsize=text_size) 
matplotlib.rc('ytick', labelsize=text_size) 

def plot_model(K, model, n0=0):
    
    D = np.load("results_GLLiM/result_model" + str(model) +"_K" + str(K) +"_n"\
                + str(int(ns[-1])) +"_rep" + str(reps)+ "_loss.npy")

    fig = plt.figure()
    
    # # Test sample sizes from 10^3-10^5 using n0=5
    # loss        = np.mean(D[5:,:], axis=1)
    # yerr        = 2*np.std(D[5:,:], axis=1)
    # lab="temp"
    
    ## Test sample sizes from 10^2-10^5
    loss        = np.mean(D, axis=1)
    yerr        = 2*np.std(D, axis=1)
    lab="temp"

    Y = np.array(np.log(loss)).reshape([-1,1])
    if ((model == 1) or (model == 3)):
        label = "$\\overline{\mathcal{D}}(\widehat G_n, G_0)$"
    else:
        label = "$\\widetilde{\mathcal{D}}(\widehat G_n, G_0)$"

    plt.errorbar(np.log(ns[n0:]), Y[n0:].reshape([-1]), yerr=yerr[n0:], color='orange', linestyle = '-', lw=lw, elinewidth=elw, label=label)
    # plt.errorbar(np.log(ns[n0:]), Y[n0:], label=label)
    # plt.grid(True, alpha=.5)

    X = np.empty([ns.size-n0, 2])
    X[:,0] = np.repeat(1, ns.size-n0)
    X[:,1] = np.log(ns[n0:])
    Y = Y[n0:] 
        
    beta = (np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y)
    print("Beta: ", beta[1,0])

    plt.plot(X[:,1], X @ beta, lw=lw, color='black', linestyle = '-.', label=str(np.round(beta[0,0], 2)) +\
             "$n^{" + str(np.round(beta[1,0],5)) + "}$" )
    
    plt.xlabel("log(sample size)", fontsize=text_size)
    plt.ylabel("log(loss)", fontsize=text_size)#"$\log$ " + lab)
    plt.legend(loc="upper right", title="", prop={'size': text_size})

    plt.savefig("plots_GLLiM/plot_model" + str(model) +"_K" + str(K) + "_n0_" +\
                str(n0) +"_n" + str(int(ns[-1])) +"_rep" + str(D.shape[1])+\
                    ".pdf", bbox_inches = 'tight',pad_inches = 0)

# print("Log-log scale plots for the simulation results of CRPE-GLLiM")
plot_model(model= model, K= K)