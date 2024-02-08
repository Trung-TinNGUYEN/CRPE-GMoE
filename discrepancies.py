import numpy as np

# For GLLiM
def gauss_dist_GLLiM(c0, Gamma0, A0, b0, nu0, c, Gamma, A, b, nu):
    return np.linalg.norm(c0 - c) + np.linalg.norm(Gamma0-Gamma) + np.linalg.norm(A0-A) +\
        np.linalg.norm(b0-b) + np.linalg.norm(nu0-nu)


rbar = [0,1,4,6] # cf. Proposition 2.1 of [Ho and Nguyen (2016), Annals of Statistics].
# Type I setting
def gauss_loss_GLLiM1(pi0, c0, Gamma0, A0, b0, nu0, pi, c, Gamma, A, b, nu):
    K0 = c0.shape[0]
    K  = c.shape[0]
    
    D = np.empty((K,K0))

    for i in range(K):
        for j in range(K0):
            D[i,j] = gauss_dist_GLLiM(c0[j,:], Gamma0[j,:,:], A0[j,:,:], b0[j,:], nu0[j,:,:],\
                                      c[i,:], Gamma[i,:,:], A[i,:,:], b[i,:], nu[i,:,:])

    vor=[]
    for i in range(K):
        for k in range(K0):
            if D[i,k] == np.min(D[i,:]):
                vor.append(k)

    unique, counts = np.unique(vor, return_counts=True)
    d = dict(zip(unique, counts))

    mask = ~np.eye(c.shape[1],dtype=bool)
    summ = 0.0
    for i in range(K):
        j = vor[i]
        if counts[vor[i]] == 1:
            summ += pi[i] * D[i,vor[i]] 

        else:
            j = vor[i]
            rb = rbar[counts[j]]
            c_dist = (np.linalg.norm(c[i,:]-c0[j,:]))**rb
            Gamma_dist = (np.linalg.norm(Gamma[i,:,:] - Gamma0[j,:,:]))**(rb/2.0)
            A_dist = (np.linalg.norm(A[i,:,:] - A0[j,:,:]))**(2.0)
            b_dist = (np.linalg.norm(b[i,:] - b0[j,:]))**rb
            nu_dist = (np.linalg.norm(nu[i,:,:] - nu0[j,:,:]))**(rb/2.0)
            summ += pi[i] * (c_dist + Gamma_dist + A_dist + b_dist + nu_dist)

    for k in range(K0):
        pi_bar = 0

        for i in range(K):
            if vor[i] == k:
                pi_bar += pi[i]

        summ += np.abs(pi_bar - pi0[k])

    return summ

## Type II setting
##  c0[j,:] = 0 for all j \in [\tilde{k}_0] 
## and c0[j,:] \neq 0 for all [\tilde{k}_0 + 1] \le j \le \in [k0].
rtilde = [0,1,4,6]
def gauss_loss_GLLiM2(pi0, c0, Gamma0, A0, b0, nu0, pi, c, Gamma, A, b, nu):
    K0 = c0.shape[0]
    K  = c.shape[0]
    
    D = np.empty((K,K0))

    for i in range(K):
        for j in range(K0):
            D[i,j] = gauss_dist_GLLiM(c0[j,:], Gamma0[j,:,:], A0[j,:,:], b0[j,:], nu0[j,:,:],\
                                      c[i,:], Gamma[i,:,:], A[i,:,:], b[i,:], nu[i,:,:])

    vor=[]
    for i in range(K):
        for k in range(K0):
            if D[i,k] == np.min(D[i,:]):
                vor.append(k)

    unique, counts = np.unique(vor, return_counts=True)
    
    summ = 0.0
    for i in range(K):
        j = vor[i]
        if counts[vor[i]] == 1:
            summ += pi[i] * D[i,vor[i]] 

        else:
            j = vor[i]
            rt = rtilde[counts[j]]
            
            if (c0[j,:] == np.zeros([1,c0.shape[1]])).all(): # c0[j,:] = 0 for all j \in [\tilde{k}_0] 
                c_dist = (np.linalg.norm(c[i,:]-c0[j,:]))**rt
                Gamma_dist = (np.linalg.norm(Gamma[i,:,:] - Gamma0[j,:,:]))**(rt/2.0)
                A_dist = (np.linalg.norm(A[i,:,:] - A0[j,:,:]))**(rt/2.0)
                b_dist = (np.linalg.norm(b[i,:] - b0[j,:]))**rt
                nu_dist = (np.linalg.norm(nu[i,:,:] - nu0[j,:,:]))**(rt/2.0)
                summ += pi[i] * (c_dist + Gamma_dist + A_dist + b_dist + nu_dist)
                
            else:
                c_dist = (np.linalg.norm(c[i,:]-c0[j,:]))**rt
                Gamma_dist = (np.linalg.norm(Gamma[i,:,:] - Gamma0[j,:,:]))**(rt/2.0)
                A_dist = (np.linalg.norm(A[i,:,:] - A0[j,:,:]))**(2.0)
                b_dist = (np.linalg.norm(b[i,:] - b0[j,:]))**rt
                nu_dist = (np.linalg.norm(nu[i,:,:] - nu0[j,:,:]))**(rt/2.0)
                summ += pi[i] * (c_dist + Gamma_dist + A_dist + b_dist + nu_dist)
                
    for k in range(K0):
        pi_bar = 0

        for i in range(K):
            if vor[i] == k:
                pi_bar += pi[i]

        summ += np.abs(pi_bar - pi0[k])

    return summ



