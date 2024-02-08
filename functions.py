import numpy as np
from copy import deepcopy
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import sys
from scipy.special import logsumexp

def sample_GLLiM(pi_true, c_true, Gamma_true, A_true, b_true, Sigma_true, nb_data, seed = 0):
    """
    Draw nb_data samples (Xi, Yi), i = 1,...,nb_data, from a supervised Gaussian locally-linear mapping (GLLiM).

    Parameters
    ----------
    pi_true : ([nb_mixture] np.array)
        Mixing proportion.
    c_true : ([nb_mixture, dim_data_X], np.array)
        Means of Gaussian components.
    Gamma_true : ([nb_mixture, dim_data_X, dim_data_X], np.array)
        Covariance matrices of Gaussian components.
    A_true : ([k_trunc, dim_data_Y, dim_data_X] np.array)
        Regressor matrix of location parameters of normal distribution.       
    b_true : ([k_trunc, dim_data_Y] np.array)
        Intercept vector of location parameters of normal distribution.       
    Sigma_true : ([k_trunc, dim_data_Y, dim_data_Y] np.array)
        Scale parameters (Gaussian experts' covariance matrices) of normal distribution.            
    nb_data : int
        Sample size.    
    seed : int
        Starting number for the random number generator.    
                    
    Returns
    -------
    data_X : ([nb_data, dim_data_X] np.array) 
        Input random sample.
    data_Y : ([nb_data, dim_data_Y] np.array)
        Output random sample.

    """
    ############################################    
    # Sample the input data: data_X
    ############################################   
     
    # Generate nb_data samples from Gaussian mixture model (GMM).

    # Draw nb_data samples from a multinomial distribution.
    rng = np.random.default_rng(seed)
    sample_multinomial = rng.multinomial(1, pi_true, size = nb_data)
    ## Test
    # N = 10
    # rvs = rng.multinomial(1, [0.2, 0.3, 0.5], size = N); rvs
    # Return the categories.
    kclass_X = sample_multinomial.argmax(axis = -1)
    #kclass

    # Draw nb_data samples from a multivariate normal distribution based on kclass.
    dim_data_X = c_true.shape[1]
    data_X = np.zeros((nb_data, dim_data_X))
    
    for n in range(nb_data):
        data_X[n, None] = rng.multivariate_normal(mean = c_true[kclass_X[n]], cov = Gamma_true[kclass_X[n]])
    
    # # Plot sample data_X. 
    # import matplotlib.pyplot as plt
    # plt.plot(data_X, data_X, 'x')
    # plt.axis('equal')
    # plt.show()

    ############################################    
    # Sample the output data: data_Y
    ############################################ 
    dim_data_Y = b_true.shape[1]
    data_Y = np.zeros((nb_data, dim_data_Y))

    # Calculate the gating network probabilites
    # Equivalent to calculation of a posteriori probas in a GMM.

    gating_prob = posterior_Gaussian_gate(data_X, pi_true, c_true, Gamma_true)[0]
    
    nb_mixture = len(pi_true)
    latent_Z =  np.zeros((nb_data, nb_mixture))
    kclass_Y =  np.zeros((nb_data, 1))
    
    for n in range(nb_data):
        Znk = rng.multinomial(1, gating_prob[n], size = 1)[0]
        latent_Z[n] = Znk
        zn = np.where(Znk == 1)[0]
        kclass_Y[n] = zn[0]
        # Sample Y
        data_Y[n, None] = b_true[zn] + (A_true[zn[0], :, :]@data_X[n, None].T).reshape(1, dim_data_Y)\
                            + rng.multivariate_normal(mean = np.zeros((dim_data_Y)),
                                                      cov = Sigma_true[zn[0], :, :])
        
    # return (data_X, data_Y, latent_Z, kclass_Y)
    return (data_X, data_Y)

def posterior_Gaussian_gate(data, weight, mean, cov):
    """
    Compute responsibilities in a Gaussian Mixture Model.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.    
    weight : ([nb_mixture], np.array)
        Mixing proportion.
    mean : ([nb_mixture, dim_data], np.array)
        Means of Gaussian components.
    cov : ([nb_mixture, dim_data, dim_data], np.array)
        Covariance matrices of Gaussian components.
        
    Returns
    -------
    respons: ([nb_data, nb_mixture], np.array)
        Responsibilities.
    log_pik_Nik : ([nb_data, nb_mixture], np.array)
          Log of product between the weights and the PDF of a multivariate GMM.  
    """
    # loglik : np.float32
    #     Log-likelihood of GMM.    
    
    nb_mixture = len(weight)
    log_pik_Nik = Gaussian_weight_pdf(data, weight, mean, cov)[0]
    log_sum_exp_pik_Nik = logsumexp_GLLiM(log_pik_Nik, 1)
    log_responsik = log_pik_Nik - log_sum_exp_pik_Nik@np.ones((1, nb_mixture))
    respons = np.exp(log_responsik)
    # loglik = np.sum(log_sum_exp_pik_Nik)
    
    return respons, log_pik_Nik

def Gaussian_weight_pdf(data, weight, mean, cov):
    """
    Calculate the log product between the weights and the PDF 
    of a Gaussian distribution.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.    
    weight : ([nb_mixture], np.array)
        Mixing proportion.
    mean : ([nb_mixture, dim_data], np.array)
        Means of Gaussian components.
    cov : ([nb_mixture, dim_data, dim_data], np.array)
        Covariance matrices of Gaussian components.
    Returns
    -------
    log_Nik : ([nb_data, nb_mixture], np.array)
        Log Pdf of a Gaussian distribution PDF.
        
    log_pik_Nik : ([nb_data, nb_mixture], np.array)
        Log of product between the weights 
        and the PDF of a Gaussian distribution.  
    """
    
    (nb_data, dim_data) = np.shape(data)
    nb_mixture = len(weight)
    log_Nik = np.zeros((nb_data, nb_mixture))
    log_pik_Nik = np.zeros((nb_data, nb_mixture))
    
    for k in range(nb_mixture):
        if dim_data == 1:
            log_Nik[:, k, None] = Gaussian_ndata(data, mean[None,k,:], cov[k])[0]
            log_pik_Nik[:, k, None] = np.ones((nb_data, 1))* np.log(weight[k]) + log_Nik[:, k, None]
            
        log_Nik[:, k, None] = Gaussian_ndata(data, mean[k,:], cov[k])[0]
        log_pik_Nik[:, k, None] = np.ones((nb_data, 1))* np.log(weight[k]) + log_Nik[:, k, None]    
            
        
    
    return log_pik_Nik, log_Nik

def Gaussian_ndata(data, mean, cov):
    """
    Calculate pdf of a Gaussian distribution over nb_data.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.
    mean : ([1, dim_data], np.array)
        Means of Gaussian.
    cov : ([dim_data, dim_data], np.array)
        Covariance matrices of Gaussian.
        
    Returns
    -------
    log_Ni : ([nb_data, 1], np.array)
        Log pdf of a Gaussian distribution.
    Ni : ([nb_data, 1], np.array)
        Pdf of a Gaussian distribution.
    """
    
    # ## Code from scratch  without using multivariate_normal.
    # (nb_data, dim_data) = np.shape(data)
    # det_cov = np.linalg.inv(cov)
    # z = ((data - np.ones((nb_data, 1))@mean)@det_cov)*(data - np.ones((nb_data, 1))@mean)
    # mahalanobis = np.sum(z, axis=1, keepdims=True)
    # log_Ni = -(dim_data/2)*np.log(2*np.pi) - 0.5*np.log(det_cov) - 0.5*mahalanobis
    # Ni = np.exp(log_Ni)
    
    (nb_data, dim_data) = np.shape(data)
    log_Ni = np.zeros((nb_data, 1))
    Ni = np.ones((nb_data, 1))
    
    if dim_data == 1:
        for n in range(nb_data):
            log_Ni[n, :, None] =\
                multivariate_normal.logpdf(data[n, :, None], mean,
                                           cov+sys.float_info.epsilon*np.diag(np.ones(dim_data)))
            Ni[n, :, None] =\
                multivariate_normal.pdf(data[n, :, None], mean,
                                        cov+cov+sys.float_info.epsilon*np.diag(np.ones(dim_data)))
    
    for n in range(nb_data):
        log_Ni[n, :] =\
            multivariate_normal.logpdf(data[n, :], mean,
                                       cov+sys.float_info.epsilon*np.diag(np.ones(dim_data)))
        Ni[n, :] =\
            multivariate_normal.pdf(data[n, :], mean,
                                    cov+cov+sys.float_info.epsilon*np.diag(np.ones(dim_data)))
    
    return log_Ni, Ni

def logsumexp_GLLiM(x, dimension):
    """
    Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
    By default dim = 1 (row).

    Parameters
    ----------
    x : np.array
        Input data.
    dimension : int
        0: Column sum 1: Row sum.

    Returns
    -------
    log_sum_exp: np.float64
        Value of log(sum(exp(x),dim)).

    """
    # Subtract the largest value in each row
    x_max = np.amax(x, dimension, keepdims=True)
    x = x - x_max
    x_log_sum_exp = x_max + np.log(np.sum(np.exp(x), dimension, keepdims=True))
    
    x_max_check_inf = np.isinf(x_max)
    if np.sum(x_max_check_inf) > 0:
        x_log_sum_exp[x_max_check_inf] = x_max[x_max_check_inf]
    
    return x_log_sum_exp

## log_gauss_densities.py
_LOG_2PI = np.log(2 * np.pi)

# log of pdf for gaussian distributuion with diagonal covariance matrix
def loggausspdf(X, mu, cov):
    if len(X.shape)==1:
        D=1
    else:
        D = X.shape[1]
    
    logDetCov = D*np.log(cov)
    dxM = X - mu
    L = np.sqrt(cov)
    xRinv = 1/L * dxM
    mahalaDx = np.sum(xRinv**2, axis=1)
    y = - 0.5 * (logDetCov + D*_LOG_2PI + mahalaDx)
    return y

def gausspdf(X, mu, cov):
    return np.exp(loggausspdf(X, mu, cov))

# log of pdf for gaussian distributuion with full covariance matrix (cholesky factorization for stability)
def chol_loggausspdf(X, mu, cov):

    D = X.shape[0]
    
    X = X - mu #DxN
    U = np.linalg.cholesky(cov + sys.float_info.epsilon).T #DxD
    Q = np.linalg.solve(U.T,X)
    q = np.sum(Q**2, axis=0)
    c = D*_LOG_2PI + 2*np.sum(np.log(np.diag(U)))
    y = -0.5 * (c + q)

    return y 

def em_GLLiM(X, Y, pi_start, c_start, Gamma_start, A_start, b_start, nu_start, max_iter=1000, eps=1e-6):

    pi_prev    = pi_start
    c_prev     = c_start
    Gamma_prev = Gamma_start
    A_prev     = A_start
    b_prev     = b_start
    nu_prev    = nu_start
 
    pi_new    = pi_start    
    c_new     = c_start
    Gamma_new = Gamma_start
    A_new     = A_start
    b_new     = b_start
    nu_new    = nu_start
    
    K = np.size(pi_start)
    n = X.shape[0]
    d = X.shape[1]
    l = Y.shape[1]
    LL= np.ndarray((max_iter,1))
    
    for iiter in range(max_iter):
        
        #  E-Z step:
        # print("E-Z")    
        logrnk =  np.ndarray((n,K))
        for k in range(K):
            y_mean = np.dot(A_new[k,:,:],X.T) + b_new[k,:].reshape((l,1))
            logrnk[:,k] = np.log(pi_new[k]) + chol_loggausspdf(X.T, c_new[k,:].reshape((d,1)), Gamma_new[k,:,:]) +  loggausspdf(Y, y_mean.T, nu_new[k,:,:])
        
        lognormrnk = logsumexp(logrnk, axis=1, keepdims=True)
        logrnk -= lognormrnk
        rnk = np.exp(logrnk)
        
        LL[iiter,0] = np.sum(lognormrnk) # EVERY EM Iteration THIS MUST INCREASE
        # LL_iiter = np.sum(lognormrnk) # EVERY EM Iteration THIS MUST INCREASE
        # print("Log-likelihood = " + str(LL[iiter,0]) + " at iteration nb :" + str(iiter))
        
        rk = np.empty([K])
        for k in range(K):
            rk[k] = np.sum(rnk[:,k])      
        
        #  M-GMM-step:
        # print("M-GMM")
        for k in range(K):
            pi_new[k] = rk[k]/n # (28)
            c_new[k,:] = np.sum(rnk[:,k]*X.T,axis=1)/rk[k] # (29)
            Gamma_new[k,:,:] = np.dot((np.sqrt(rnk[:,k]).reshape((1,n)))*(X.T-c_new[k,:].reshape((d,1))),\
                                        ((np.sqrt(rnk[:,k]).reshape((1,n)))*(X.T-c_new[k,:].reshape((d,1)))).T)/rk[k] # (30)
        
        # M-mapping-step
        # print("M-mapping")
        xk_bar = np.empty([K,d])
        yk_bar = np.empty([K,l])
        for k in range(K):
            xk_bar[k,:] = np.sum(rnk[:,k]*X.T,axis=1)/rk[k] # (35)
            yk_bar[k,:] = np.sum(rnk[:,k]*Y.T,axis=1)/rk[k] # (36)
        
        XXt_stark = np.zeros((d,d))
        YXt_stark = np.zeros((l,d))
        
        for k in range(K):
            X_stark = (np.sqrt(rnk[:,k]))*(X-xk_bar[k,:]).T # (33)
            Y_stark = (np.sqrt(rnk[:,k]))*(Y-yk_bar[k,:]).T  # (34)
            XXt_stark = np.dot(X_stark,X_stark.T)
            # XXt_stark += sys.float_info.epsilon*np.diag(np.ones(XXt_stark.shape[0])) # Numerical stability
            
            YXt_stark = np.dot(Y_stark,X_stark.T)        
            A_new[k,:,:] = np.dot(YXt_stark,inv(XXt_stark)) # (31)
        
        for k in range(K):    
            b_new[k,:] = np.sum(rnk[:,k].T*(Y-(A_new[k,:,:].dot(X.T)).T).T,axis=1)/rk[k] # (37)
        
        for k in range(K):
            nu_new[k,:,:] = np.dot((np.sqrt(rnk[:,k]).reshape((1,n)))*(Y.T-A_new[k,:,:].dot(X.T)-b_new[k,:].reshape((l,1))),\
                                        ((np.sqrt(rnk[:,k]).reshape((1,n)))*(Y.T-A_new[k,:,:].dot(X.T)-b_new[k,:].reshape((l,1)))).T)/rk[k] # (38)
            
        # print('nu_new = ', nu_new)
        
        ## Stoping criterion.
        if (iiter >= 3):
            deltaLL_total = np.amax(LL[0:iiter,0])-np.amin(LL[0:iiter,0])
            deltaLL = LL[iiter,0]-LL[iiter-1,0]  # EVERY EM Iteration THIS MUST INCREASE
            converged = bool(deltaLL <= eps*deltaLL_total)        
        
        if max(np.linalg.norm(c_new - c_prev), np.linalg.norm(Gamma_new - Gamma_prev),\
            np.linalg.norm(A_new - A_prev), np.linalg.norm(b_new - b_prev),\
                np.linalg.norm(nu_new - nu_prev))< eps or (iiter > max_iter) or converged:
            break
        
        # if iiter > 500 or (iiter > 100 and iiter % 10 == 0):
        #    if (np.linalg.norm(c_new - c_prev) + np.linalg.norm(Gamma_new - Gamma_prev) +\
        #        np.linalg.norm(A_new - A_prev) + np.linalg.norm(b_new - b_prev) +\
        #            np.linalg.norm(nu_new - nu_prev))< eps or (iiter > max_iter) or converged:
        #        break
        
        pi_prev    = deepcopy(pi_new)
        c_prev     = deepcopy(c_new)
        Gamma_prev = deepcopy(Gamma_new)
        A_prev     = deepcopy(A_new)
        b_prev     = deepcopy(b_new)
        nu_prev    = deepcopy(nu_new)
        
    return (pi_new, c_new, Gamma_new, A_new, b_new, nu_new, iiter, LL)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)