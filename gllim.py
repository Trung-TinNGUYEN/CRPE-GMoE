"""
The equation numbers refer to (Deleforge 2015, Statistics and Computing). 
"High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables".
"""

import numpy as np
from numpy.linalg import inv
import sys
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from itertools import compress
import time
from log_gauss_densities import loggausspdf,chol_loggausspdf

class GLLiM:
    ''' Gaussian Locally-Linear Mapping'''
    
    def __init__(self, K_in, D_in, L_in, N_in, random_state):

        self.K = K_in
        self.D = D_in
        self.L = L_in
        self.N = N_in

        ## Make a partition of starting values near the true components. 
        ## TBC using init_params_GLLiM(n, K)
        ## Inverse regression (used for training)
        ## Set arbitrary initial values.
        np.random.seed(random_state)

        self.pikList = [1.0/self.K for k in range(self.K)]        
        self.ckList = [np.random.randn(self.L,1) for k in range(self.K)]
        self.GammakList = [np.ones((self.L, self.L)) for k in range(self.K)]
        self.AkList = [np.random.randn(self.D, self.L) for k in range(self.K)]
        self.bkList = [np.random.randn(self.D,1) for k in range(self.K)]
        self.SigmakSquareList = [np.ones((self.D,1)) for k in range(self.K)] #isotropic sigma.
        self.rnk = np.empty((self.N,self.K)) # need to be allocated
         
    def fit(self, X, Y, max_iter, true_init = None, random_state = None):
        '''fit the Gllim
           # Arguments
            X: low dimension targets as a Numpy array
            Y: high dimension features as a Numpy array
            max_iter: maximum number of EM algorithm iterations
            init: boolean, compute GaussianMixture initialisation
            gmm_init: give a GMM as init
        '''

        N = X.shape[0]
        LL=np.ndarray((max_iter,1))
        it = 0
        converged = False
        
        start_time_EM = time.time()
        
        if true_init == True:
            
            #  E-Z step-0:
            logrnk = np.ndarray((N,self.K))
            # print("E-Z-0")
            for (k, Ak, bk, ck, pik, gammak, sigmakSquare) in zip(range(self.K), self.AkList, self.bkList, self.ckList, self.pikList, self.GammakList, self.SigmakSquareList):
                
                y_mean = np.dot(Ak,X.T) + bk.reshape((self.D,1))
                logrnk[:,k] = np.log(pik) + chol_loggausspdf(X.T, ck.reshape((self.L,1)), gammak) +  loggausspdf(Y, y_mean.T, sigmakSquare)
            
            lognormrnk = logsumexp(logrnk,axis=1,keepdims=True)
            logrnk -= lognormrnk
            self.rnk = np.exp(logrnk)
            rkList = [np.sum(self.rnk[:,k]) for k in range(self.K)]
            
        else:
            # print("Start initialization")
            deltaLL = float('inf')
            # print("X : ", X.shape)
            # print("Y : ", Y.shape)
    
            ## We initialize the model by running a GMM on the input only
            datas_matrix = X
    
            ## Uncomment the following line if you want to initialize the model on the complete data
            # datas_matrix = np.concatenate((X, Y), axis=1) #complete data matrix
    
            # print("datas matrix shape:", datas_matrix.shape)
            
            # print("Initialization of posterior with GMM")
            # start_time_EMinit = time.time()
            
            gmm = GaussianMixture(n_components=self.K, covariance_type='diag', 
                                  random_state = random_state, tol=0.001, n_init=5,
                                  init_params='k-means++', verbose=0)
            gmm.fit(datas_matrix)
            
            self.rnk = gmm.predict_proba(datas_matrix)
            rkList = [np.sum(self.rnk[:,k]) for k in range(self.K)]
            
            logrnk = np.ndarray((N,self.K))                
            
            # print("--- %s seconds for EM initialization---" % (time.time() - start_time_EMinit))
            
        print("Training with GLLiM-EM")
        # start_time_EM = time.time()
        
        while (converged==False) and (it<max_iter):

            it += 1 

            print("Iteration nb "+str(it))
            
            #  M-GMM-step:
            # print("M-GMM")
            
            self.pikList=[rk/N for rk in rkList] # (28)

            self.ckList=[np.sum(self.rnk[:,k]*X.T,axis=1)/rk for k,rk in enumerate(rkList)] # (29)

            self.GammakList=[np.dot((np.sqrt(self.rnk[:,k]).reshape((1,N)))*\
                                    (X.T-ck.reshape((self.L,1))),\
                                        ((np.sqrt(self.rnk[:,k]).reshape((1,N)))*\
                                         (X.T-ck.reshape((self.L,1)))).T)/rk \
                             for k,ck,rk in zip(range(self.K),self.ckList,rkList)]  # (30)

            # M-mapping-step
            # print("M-mapping")
            xk_bar = [np.sum(self.rnk[:,k]*X.T,axis=1)/rk for k,rk in enumerate(rkList)]# (35)
            
            yk_bar = [np.sum(self.rnk[:,k]*Y.T,axis=1)/rk for k,rk in enumerate(rkList)]  # (36)

            XXt_stark=np.zeros((self.L,self.L))
            YXt_stark=np.zeros((self.D,self.L))

            for k,rk,xk,yk in zip(range(self.K),rkList,xk_bar,yk_bar):
                
                X_stark=(np.sqrt(self.rnk[:,k]))*(X-xk).T  # (33)
                Y_stark=(np.sqrt(self.rnk[:,k]))*(Y-yk).T  # (34)
                ## TestT
                XXt_stark=np.dot(X_stark,X_stark.T)
                XXt_stark+=sys.float_info.epsilon*np.diag(np.ones(XXt_stark.shape[0]))
                YXt_stark=np.dot(Y_stark,X_stark.T)                
                self.AkList[k]=np.dot(YXt_stark,inv(XXt_stark))
            
            self.bkList=[np.sum(self.rnk[:,k].T*(Y-(Ak.dot(X.T)).T).T,axis=1)/rk \
                         for k,Ak,rk in zip(range(self.K),self.AkList ,rkList)]  # (37)
            
            diffSigmakList = [np.sqrt(self.rnk[:,k]).T*(Y-(Ak.dot(X.T)).T-bk.reshape((1,self.D))).T \
                              for k,Ak,bk in zip(range(self.K),self.AkList,self.bkList)]
           
            sigma2 = [np.sum((diffSigma**2),axis=1)/rk for rk,diffSigma in zip(rkList,diffSigmakList)]
            
            # print('sigma2 = ', sigma2)
            
            self.SigmakSquareList = [(np.sum(sig2)/self.D) for sig2 in sigma2] 
            print('sigma2_iso = ', self.SigmakSquareList)
            
            ## Numerical stability
            self.SigmakSquareList = [sig + sys.float_info.epsilon for sig in self.SigmakSquareList] 
        
            #  E-Z step:
            # print("E-Z")
            for (k, Ak, bk, ck, pik, gammak, sigmakSquare) in \
                zip(range(self.K), self.AkList, self.bkList, self.ckList,\
                    self.pikList, self.GammakList, self.SigmakSquareList):
                
                y_mean = np.dot(Ak,X.T) + bk.reshape((self.D,1))
                logrnk[:,k] = np.log(pik) + \
                    chol_loggausspdf(X.T, ck.reshape((self.L,1)), gammak) +\
                        loggausspdf(Y, y_mean.T, sigmakSquare)

            lognormrnk = logsumexp(logrnk,axis=1,keepdims=True)
            logrnk -= lognormrnk
            self.rnk = np.exp(logrnk)
            
            LL[it,0] = np.sum(lognormrnk) # EVERY EM Iteration THIS MUST INCREASE
            print("Log-likelihood = " + str(LL[it,0]) + " at iteration nb :" + str(it))
            rkList=[np.sum(self.rnk[:,k]) for k in range(self.K)]
            
            # ## Remove empty clusters. CRPE-GLLiM.
            # ec = [True]*self.K
            # cpt = 0
            # for k in range(self.K):
            #     if (np.sum(self.rnk[:,k])==0) or (np.isinf(np.sum(self.rnk[:,k]))):
            #         cpt +=1
            #         ec[k] = False
            #         print("class ",k," has been removed")
            # self.K -= cpt
            # rkList = list(compress(rkList, ec))
            # self.AkList = list(compress(self.AkList, ec))
            # self.bkList = list(compress(self.bkList, ec))
            # self.ckList = list(compress(self.ckList, ec))
            # self.SigmakSquareList = list(compress(self.SigmakSquareList, ec))
            # self.pikList = list(compress(self.pikList, ec))
            # self.GammakList = list(compress(self.GammakList, ec))
            
            if (it>=3):
                deltaLL_total = np.amax(LL[1:it,0])-np.amin(LL[1:it,0])
                deltaLL = LL[it,0]-LL[it-1,0]
                converged = bool(deltaLL <= (1e-3)*deltaLL_total)
                        
        print("Final log-likelihood : " + str(LL[it,0]))

        print(" Converged in %s iterations" % (it))

        print("--- %s seconds for EM ---" % (time.time() - start_time_EM))

        # plt.plot(LL[1:it,0])
        # plt.show()
        # pikList = self.pikList
        
        return (self.pikList, self.ckList, self.GammakList, self.AkList,\
                self.bkList, self.SigmakSquareList, it, LL[1:it,0])
