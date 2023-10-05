from inputlds import*
from ncpol2sdpa import*
import numpy as np
import pandas as pd
from math import sqrt
import random


def multiple_SimCom(Y,T,level):
# Define a function for solving the NCPO problems with 
# given standard deviations of process noise and observtion noise,
# length of  estimation data and required relaxation level. 

    # Decision Variables
    G = generate_operators("G", n_vars=1, hermitian=True, commutative=False)[0]
    Fdash = generate_operators("Fdash", n_vars=1, hermitian=True, commutative=False)[0]
    m = generate_operators("m", n_vars=T+1, hermitian=True, commutative=False)
    q = generate_operators("q", n_vars=T, hermitian=True, commutative=False)
    p = generate_operators("p", n_vars=T, hermitian=True, commutative=False)
    f = generate_operators("f", n_vars=T, hermitian=True, commutative=False)
    
    # Objective
    obj = sum(sum(np.inner(Y[i,j]-f[j], Y[i,j]-f[j]) for j in range(T)) for i in range(Y.shape[0])) + 0.0005*sum(p[i]**2 for i in range(T)) + 0.0001*sum(q[i]**2 for i in range(T))

    # Constraints
    ine1 = [f[i] - Fdash*m[i+1] - p[i] for i in range(T)]
    ine2 = [-f[i] + Fdash*m[i+1] + p[i] for i in range(T)]
    ine3 = [m[i+1] - G*m[i] - q[i] for i in range(T)]
    ine4 = [-m[i+1] + G*m[i] + q[i] for i in range(T)]
    #ine5 = [(Y[i]-f[i])**2 for i in range(T)]
    ines = ine1+ine2+ine3+ine4 #+ine5

    
    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,Fdash,f,p,m,q]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, inequalities=ines)
    sdp.solve(solver='mosek')
    #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
    #print(sdp.primal, sdp.dual, sdp.status)
    if (sdp[sum(sum(np.inner(Y[i,j]-f[j], Y[i,j]-f[j]) for j in range(T)) for i in range(Y.shape[0])) ] < 0):
        print("sum(sum(np.inner(Y[i,j]-f[j], Y[i,j]-f[j]) for j in range(T)) for i in range(Y.shape[0]))  < 0")
        return 

    
    #the first is nmrse with sum of all the trajectories which will probably go to 0 no matter the performance (the fraction is getting to 1 since both numerator and denominator are getting higher)
    #the second is only for 0th trajectory (and we can compute it for each trajectory)
    #nrmse_sim = 1-sqrt(sdp[sum(sum(np.inner(Y[i,j]-f[j] + q[j], Y[i,j]-f[j] + q[j]) for j in range(T)) for i in range(Y.shape[0])) ])/sqrt(sum(sum(np.inner(Y[i,j]- Y.mean(axis=(0,1)), Y[i,j]- Y.mean(axis=(0,1))) for j in range(T)) for i in range(Y.shape[0])))
    nrmse_sim = 1-sqrt(sdp[sum(np.inner(Y[0,j]-f[j] + q[j], Y[0,j]-f[j] + q[j]) for j in range(T)) ])/sqrt(sum(np.inner(Y[0,j]- Y[0].mean(axis=(0)), Y[0,j]- Y[0].mean(axis=(0))) for j in range(T)))

    #nrmse_sim = 1-sqrt(sdp[sum((Y[0,i]-f[i]+ q[i])**2 for i in range(T))])/sqrt(sum((Y[0,i]-np.mean(Y[0]))**2 for i in range(T)))

    print("G ", sdp[G])
    print("fdash ", sdp[Fdash])
    
    for i in range(T):
       print("q",i, " ", sdp[q[i]])
    
    for i in range(T):
       print("p",i, " ", sdp[p[i]])

    for i in range(T):
       print("f",i, " ", sdp[f[i]])
    
    for i in range(T + 1):
       print("m",i, " ", sdp[m[i]])

    for i in range(1,T + 1):
       print("fdas*m ",i, " ", sdp[Fdash*m[i+1]])
       

    if(sdp.status != 'infeasible'):
        return nrmse_sim, sdp
    else:
        print('Cannot find feasible solution.')
        return


def Kalman_filter(G, F, W, V, trajectory):

  #algorithm taken from https://github.com/jmarecek/OnlineLDS/blob/master/onlinelds.py and altered to 2D data
  
  T = trajectory.shape[0]
  n = G.shape[0]

  Id = np.eye(n)
  if F.shape[1] == 1:
     m_prev = 1 #np.matrix([[0],[0]]), originally 0
  else:
     m_prev = np.array([1,1]) #originally [0,0]
  
  C_prev = np.zeros((n,n))

  y_pred_full = [ 0 ]
  pred_error = [ np.linalg.norm(trajectory[0]) ]
    
  for t in range(1,T):    
    a = np.dot(G,m_prev)
    R = np.dot(G,np.dot(C_prev,G.T)) + W
    
    f = np.dot(F.T,a)    
    RF = np.dot(R,F)
    Q = np.dot(F.T,RF) + V
    A = RF
    try: A = np.dot(RF, np.linalg.inv(Q))
    except: print("Zero Q? Check %s" % str(Q))
    
    #thats on purpose in a bit slower form, to test the equations
    y_pred = np.dot(F.T, np.dot(G,m_prev))
    #print("A: ", A.shape, " R:", R.shape, "RF: ", RF.shape, " F.T: ", F.T.shape, " a:", a.shape, "y_pred: ", y_pred.shape)#, "Q: ", Q.shape, "V: ", V.shape
    m_prev = np.dot(A, trajectory[t]) + np.dot((Id - np.dot(A,F.T)),a)
    if Q.shape == (1,1):
        C_prev = R - Q[0,0] * np.dot(A,A.T) #different dimensional choice for Q will be needed
    else:
        C_prev = R -  np.dot(A,np.dot(Q,A.T))

    y_pred_full.append(y_pred)
    loss = pow(np.linalg.norm(trajectory[t] - y_pred), 2)
    pred_error.append(loss)
  
  return y_pred_full, pred_error

def generate_C0_data(num_trajectories, T):
   mean_C0 = (0.1, 0.2)
   cov_C0 = [[0.001, 0], [0, 0.001]] #initial: [[1, 0], [0, 1]]

   trajectories = []

   for _ in range(num_trajectories):
    trajectories.append(np.random.multivariate_normal(mean_C0, cov_C0, size=T))
   trajectories = np.array(trajectories)

   return trajectories

def generate_C1_data(num_trajectories, T):
   T_1 = 100000 #ns
   mean_C1 = (0.8, 0.1)
   cov_C1 = [[0.006, 0], [0, 0.006]] #initial: [[0.6, 0], [, 0.6]]
   mean_C0 = (0.1, 0.2)
   cov_C0 = [[0.001, 0], [0, 0.001]] #initial: [[1, 0], [0, 1]]

   classes = [0,1]
   trajectories = []

   for _ in range(num_trajectories):
      is_T1 = 1
      trajectory = []
      for point in range(T):
         is_T1 = random.choices(classes, weights = [1 - math.exp(-((point+1) * 8)/ T_1), math.exp(-((point+1) * 8)/ T_1)])#8ns is sampling speed of the device
         if is_T1 == [1]:
            IQ_pt = np.random.multivariate_normal(mean_C1, cov_C1, size=1)
            trajectory.append(IQ_pt[0])
                
         else:
            remaining = int(T - point)
            IQ_pts = np.random.multivariate_normal(mean_C0, cov_C0, size=remaining)
            for j in IQ_pts[:]:
                trajectory.append(j)
            break
         
      trajectories = list(trajectories)
      trajectories.append(trajectory)
      trajectories = np.array(trajectories)

   return trajectories

def labelled_decission(C0_losses, C1_losses):
   C0_loss = sum(C0_losses[2:])
   C1_loss = sum(C1_losses[2:])
   if C0_loss > C1_loss:
      decission = 1
      #print("measurement result: 1")
   else:
      decission = 0
      #print("measurement result: 0")
    
   return decission


def data_generation(g,f_dash,proc_noise_std,obs_noise_std,T):
# Generate Dynamic System ds1
    dim=len(g)
    ds1 = dynamical_system(g,np.zeros((dim,1)),f_dash,np.zeros((1,1)),
          process_noise='gaussian',
          observation_noise='gaussian', 
          process_noise_std=proc_noise_std, 
          observation_noise_std=obs_noise_std)
    h0= np.ones(ds1.d)
    inputs = np.zeros(T)
    ds1.solve(h0=h0, inputs=inputs, T=T)    
    return np.asarray(ds1.outputs).reshape(-1).tolist()

def SimCom_elementwise(Y,T,level):
# Define a function for solving the NCPO problems with 
# given standard deviations of process noise and observtion noise,
# length of  estimation data and required relaxation level. 

    # Decision Variables
    G = generate_variables("G", n_vars=4, commutative=True)
    Fdash = generate_variables("Fdash", n_vars=2, commutative=True)
    m = generate_variables("m", n_vars=2*(T+1), commutative=True)
    q = generate_variables("q", n_vars=2*T, commutative=True)
    p = generate_variables("p", n_vars=T, commutative=True)
    f = generate_variables("f", n_vars=T, commutative=True)

    # Objective
    obj = sum(sum((Y[i,j]-f[j])**2 for j in range(T)) for i in range(Y.shape[0])) + 0.0005*sum(p[i]**2 for i in range(T)) + 0.0001*sum(q[i]**2 for i in range(2*T))

    eqs1 = [f[i] - Fdash[0]*m[2*i+2] - Fdash[1]*m[2*i+3] - p[i] for i in range(T)]
    eqs2 = [m[i+2] - G[0]*m[i] - G[1]*m[i+1] - q[i] for i in range(0, 2*T-1, 2)]
    eqs3 = [m[i+2] - G[2]*m[i-1] - G[3]*m[i] - q[i] for i in range(1, 2*T, 2)]
    eqs = eqs1 + eqs2 + eqs3

    """
    subs = {}
    for i in range(T):
       subs[f[i]] = Fdash[0]*m[2*i+2] + Fdash[1]*m[2*i+3] + p[i]

    for i in range(0, 2*T-1, 2):
       subs[m[i+2]] = G[0]*m[i] + G[1]*m[i+1] + q[i]

    for i in range(1, 2*T, 2):
       subs[m[i+2]] = G[2]*m[i-1] + G[3]*m[i] + q[i]

    print(subs)   
    
    # Constraints
    ine1 = [f[i] - Fdash[0]*m[2*i+2] - Fdash[1]*m[2*i+3] - p[i] for i in range(T)]
    ine2 = [-f[i] + Fdash[0]*m[2*i+2] + Fdash[1]*m[2*i+3] + p[i] for i in range(T)]
    ine3 = [m[i+2] - G[0]*m[i] - G[1]*m[i+1] - q[i] for i in range(0, 2*T-1, 2)]
    ine4 = [m[i+2] - G[2]*m[i-1] - G[3]*m[i] - q[i] for i in range(1, 2*T, 2)]
    ine5 = [-m[i+2] + G[0]*m[i] + G[1]*m[i+1] + q[i] for i in range(0, 2*T-1, 2)]
    ine6 = [-m[i+2] + G[2]*m[i-1] + G[3]*m[i] + q[i] for i in range(1, 2*T, 2)]
    #ine5 = [(Y[i]-f[i])**2 for i in range(T)]
    ines = ine1+ine2+ine3+ine4+ine5+ine6
    """

    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,Fdash,f,p,m,q]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, equalities=eqs) # removeequalities=True doesn't work 
    sdp.solve(solver='mosek')
    #print("sdp: ", sdp)
    #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
    #print(sdp.primal, sdp.dual, sdp.status)
    if (sdp[sum(sum(np.inner(Y[i,j]-f[j], Y[i,j]-f[j]) for j in range(T)) for i in range(Y.shape[0])) ] < 0):
        print("sum((Y[i]-f[i])**2 for i in range(T)) < 0")
        return 
    

    nrmse_sim = 1-sqrt(sdp[sum(np.inner(Y[0,j]-f[j] + q[j], Y[0,j]-f[j] + q[j]) for j in range(T)) ])/sqrt(sum(np.inner(Y[0,j]- Y[0].mean(axis=(0)), Y[0,j]- Y[0].mean(axis=(0))) for j in range(T)))
    #nrmse_sim = 1-sqrt(sdp[sum((Y[i]-f[i])**2 for i in range(T))])/sqrt(sum((Y[i]-np.mean(Y))**2 for i in range(T)))

    G_mat = np.array([[sdp[G[0]], sdp[G[1]]], [sdp[G[2]], sdp[G[3]]]])
    Fdash_mat = np.array([[sdp[Fdash[0]], sdp[Fdash[1]]]])

    with np.printoptions(threshold=np.inf):
        print("x_mat: ", sdp.x_mat[0][0])

    with open("monoms.txt","w") as file:
       for i in range(4):
          file.write("G%d %s\n"%(i, str(sdp[G[i]])))
       for i in range(2):
          file.write("Fdash%d %s\n"%(i, str(sdp[Fdash[i]])))
       for i in range(2*T):
           file.write("q%d %s\n"%(i, str(sdp[q[i]])))
       for i in range(T):
           file.write("p%d %s\n"%(i, str(sdp[p[i]])))
       for i in range(2*T + 2):
           file.write("m%d %s\n"%(i, str(sdp[m[i]])))
       for i in range(T):
           file.write("f%d %s\n"%(i, str(sdp[f[i]])))
       for i in range(T):
           file.write("Fdash*m sum %d %s\n"%(i, str(sdp[Fdash[0]*m[2*i+2]] + sdp[Fdash[1]*m[2*i+3]])))
        

    for i in range(2*T):
       print("q",i, " ", sdp[q[i]])
    
    for i in range(T):
       print("p",i, " ", sdp[p[i]])

    for i in range(T):
       print("f",i, " ", sdp[f[i]])
    
    for i in range(2*T + 2):
       print("m",i, " ", sdp[m[i]])

    for i in range(T):
       print("fdas*m ",i, " ", sdp[Fdash[0]*m[2*i+2] + Fdash[1]*m[2*i+3]])

    sdp.write_to_file("sdp_file.csv")
    sdp.save_monomial_index("monomials.txt")

    if(sdp.status != 'infeasible'):
        print(nrmse_sim)
        return nrmse_sim, G_mat, Fdash_mat
    else:
        print('Cannot find feasible solution.')
        return
    
def SimCom_elementwise_rand(Y,T,level, pro, obs):
# Define a function for solving the NCPO problems with 
# given standard deviations of process noise and observtion noise,
# length of  estimation data and required relaxation level. 

#here I put out the noise realizations as variables since the algorithm tends to make matrices zero and always makes the noise exactly equal to datapoint of the trajectory


    # Decision Variables
    G = generate_variables("G", n_vars=4, commutative=True)
    Fdash = generate_variables("Fdash", n_vars=2, commutative=True)
    m = generate_variables("m", n_vars=2*(T+1), commutative=True)
    f = generate_variables("f", n_vars=T, commutative=True)

    # Objective
    obj = sum(sum((Y[i,j]-f[j])**2 for j in range(T)) for i in range(Y.shape[0])) #+ 0.0005*sum(p[i]**2 for i in range(T)) + 0.0001*sum(q[i]**2 for i in range(2*T))

    
    eqs1 = [f[i] - Fdash[0]*m[2*i+2] - Fdash[1]*m[2*i+3] - np.random.normal(0, scale = obs) for i in range(T)]
    eqs2 = [m[i+2] - G[0]*m[i] - G[1]*m[i+1] - np.random.normal(0, scale = pro) for i in range(0, 2*T-1, 2)]
    eqs3 = [m[i+2] - G[2]*m[i-1] - G[3]*m[i] - np.random.normal(0, scale = pro) for i in range(1, 2*T, 2)]
    eqs = eqs1 + eqs2 + eqs3
    print("eqs: ", eqs)

    """
    subs = {}
    for i in range(T):
       subs[f[i]] = Fdash[0]*m[2*i+2] + Fdash[1]*m[2*i+3] + p[i]

    for i in range(0, 2*T-1, 2):
       subs[m[i+2]] = G[0]*m[i] + G[1]*m[i+1] + q[i]

    for i in range(1, 2*T, 2):
       subs[m[i+2]] = G[2]*m[i-1] + G[3]*m[i] + q[i]

    print(subs)   
    
    # Constraints - nutno zaridit, aby u rovnosti byl stejnej seed
    ine1 = [f[i] - Fdash[0]*m[2*i+2] - Fdash[1]*m[2*i+3] - np.random.normal(0, scale = 0.1) for i in range(T)]
    ine2 = [-f[i] + Fdash[0]*m[2*i+2] + Fdash[1]*m[2*i+3] + np.random.normal(0, scale = 0.1) for i in range(T)]
    ine3 = [m[i+2] - G[0]*m[i] - G[1]*m[i+1] - np.random.normal(0, scale = 0.2) for i in range(0, 2*T-1, 2)]
    ine4 = [m[i+2] - G[2]*m[i-1] - G[3]*m[i] - np.random.normal(0, scale = 0.2) for i in range(1, 2*T, 2)]
    ine5 = [-m[i+2] + G[0]*m[i] + G[1]*m[i+1] + np.random.normal(0, scale = 0.2) for i in range(0, 2*T-1, 2)]
    ine6 = [-m[i+2] + G[2]*m[i-1] + G[3]*m[i] + np.random.normal(0, scale = 0.2) for i in range(1, 2*T, 2)]
    #ine5 = [(Y[i]-f[i])**2 for i in range(T)]
    ines = ine1+ine2+ine3+ine4+ine5+ine6
    """

    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,Fdash,f,m]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, equalities=eqs) # removeequalities=True doesn't work 
    sdp.solve(solver='mosek')
    #print("sdp: ", sdp)
    #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
    #print(sdp.primal, sdp.dual, sdp.status)
    print(sdp.x_mat[0].shape)
    if (sdp[sum(sum(np.inner(Y[i,j]-f[j], Y[i,j]-f[j]) for j in range(T)) for i in range(Y.shape[0])) ] < 0):
        print("sum((Y[i]-f[i])**2 for i in range(T)) < 0")
        return 
    

    nrmse_sim = 1-sqrt(sdp[sum(np.inner(Y[0,j]-f[j], Y[0,j]-f[j]) for j in range(T)) ])/sqrt(sum(np.inner(Y[0,j]- Y[0].mean(axis=(0)), Y[0,j]- Y[0].mean(axis=(0))) for j in range(T)))
    #nrmse_sim = 1-sqrt(sdp[sum((Y[i]-f[i])**2 for i in range(T))])/sqrt(sum((Y[i]-np.mean(Y))**2 for i in range(T)))

    G_mat = np.array([[sdp[G[0]], sdp[G[1]]], [sdp[G[2]], sdp[G[3]]]])
    Fdash_mat = np.array([[sdp[Fdash[0]], sdp[Fdash[1]]]])

    with np.printoptions(threshold=np.inf):
        print("x_mat: ", sdp.x_mat[0][0])

    for i in range(T):
       print("f",i, " ", sdp[f[i]])

    for i in range(2*T + 2):
       print("m",i, " ", sdp[m[i]])
    
    print("process eq: ", sdp[m[3] - G[2]*m[0] - G[3]*m[1]])
    print("measurement eq", sdp[f[0] - Fdash[0]*m[2] - Fdash[1]*m[3]])
    print("different values: ", sdp[Fdash[0]]*sdp[m[2]] + sdp[Fdash[1]]*sdp[m[3]], sdp[Fdash[0]*m[2] + Fdash[1]*m[3]])
    print("a ještě: ", Y[0,0]-sdp[f[0]])

    if(sdp.status != 'infeasible'):
        print(nrmse_sim)
        return nrmse_sim, G_mat, Fdash_mat
    else:
        print('Cannot find feasible solution.')
        return


def SimCom_elementwise_rand_uncons(Y,T,level, pro, obs):
# Define a function for solving the NCPO problems with 
# given standard deviations of process noise and observtion noise,
# length of  estimation data and required relaxation level. 

    # Decision Variables
    G = generate_variables("G", n_vars=4, commutative=True)
    Fdash = generate_variables("Fdash", n_vars=2, commutative=True)
    #m = generate_variables("m", n_vars=2*(T+1), commutative=True)
    #f = generate_variables("f", n_vars=T, commutative=True)

    states, observations = unconst_recursion(G, Fdash, T, pro, obs)
    print("states: ", states)
    print("observations: ", observations)

    # Objective
    obj = sum(sum((Y[i,j]-observations[j])**2 for j in range(T)) for i in range(Y.shape[0])) #+ 0.0005*sum(p[i]**2 for i in range(T)) + 0.0001*sum(q[i]**2 for i in range(2*T))

    """
    eqs1 = [f[i] - Fdash[0]*m[2*i+2] - Fdash[1]*m[2*i+3] - np.random.normal(0, scale = obs) for i in range(T)]
    eqs2 = [m[i+2] - G[0]*m[i] - G[1]*m[i+1] - np.random.normal(0, scale = pro) for i in range(0, 2*T-1, 2)]
    eqs3 = [m[i+2] - G[2]*m[i-1] - G[3]*m[i] - np.random.normal(0, scale = pro) for i in range(1, 2*T, 2)]
    eqs = eqs1 + eqs2 + eqs3
    print("eqs: ", eqs)

    
    subs = {}
    for i in range(T):
       subs[f[i]] = Fdash[0]*m[2*i+2] + Fdash[1]*m[2*i+3] + p[i]

    for i in range(0, 2*T-1, 2):
       subs[m[i+2]] = G[0]*m[i] + G[1]*m[i+1] + q[i]

    for i in range(1, 2*T, 2):
       subs[m[i+2]] = G[2]*m[i-1] + G[3]*m[i] + q[i]

    print(subs)   
    
    # Constraints - nutno zaridit, aby u rovnosti byl stejnej seed
    ine1 = [f[i] - Fdash[0]*m[2*i+2] - Fdash[1]*m[2*i+3] - np.random.normal(0, scale = 0.1) for i in range(T)]
    ine2 = [-f[i] + Fdash[0]*m[2*i+2] + Fdash[1]*m[2*i+3] + np.random.normal(0, scale = 0.1) for i in range(T)]
    ine3 = [m[i+2] - G[0]*m[i] - G[1]*m[i+1] - np.random.normal(0, scale = 0.2) for i in range(0, 2*T-1, 2)]
    ine4 = [m[i+2] - G[2]*m[i-1] - G[3]*m[i] - np.random.normal(0, scale = 0.2) for i in range(1, 2*T, 2)]
    ine5 = [-m[i+2] + G[0]*m[i] + G[1]*m[i+1] + np.random.normal(0, scale = 0.2) for i in range(0, 2*T-1, 2)]
    ine6 = [-m[i+2] + G[2]*m[i-1] + G[3]*m[i] + np.random.normal(0, scale = 0.2) for i in range(1, 2*T, 2)]
    #ine5 = [(Y[i]-f[i])**2 for i in range(T)]
    ines = ine1+ine2+ine3+ine4+ine5+ine6
    """

    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,Fdash]),verbose = 1)
    sdp.get_relaxation(level, objective=obj)
    sdp.solve(solver='mosek')

    #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
    if (sdp[sum(sum(np.inner(Y[i,j]-observations[j], Y[i,j]-observations[j]) for j in range(T)) for i in range(Y.shape[0])) ] < 0):
        print("sum((Y[i]-f[i])**2 for i in range(T)) < 0")
        return 
    

    nrmse_sim = 1-sqrt(sdp[sum(np.inner(Y[0,j]-observations[j], Y[0,j]-observations[j]) for j in range(T)) ])/sqrt(sum(np.inner(Y[0,j]- Y[0].mean(axis=(0)), Y[0,j]- Y[0].mean(axis=(0))) for j in range(T)))
    #nrmse_sim = 1-sqrt(sdp[sum((Y[i]-f[i])**2 for i in range(T))])/sqrt(sum((Y[i]-np.mean(Y))**2 for i in range(T)))

    G_mat = np.array([[sdp[G[0]], sdp[G[1]]], [sdp[G[2]], sdp[G[3]]]])
    Fdash_mat = np.array([[sdp[Fdash[0]], sdp[Fdash[1]]]])

    with np.printoptions(threshold=np.inf):
        print("x_mat: ", sdp.x_mat[0][0])

    if(sdp.status != 'infeasible'):
        print(nrmse_sim)
        return nrmse_sim, G_mat, Fdash_mat
    else:
        print('Cannot find feasible solution.')
        return
    
def unconst_recursion(G, Fdash, runs, pro, obs):
   
   if runs == 0:
      states = []
      observations = []
      state = [0, 0]
      state[0] = 1 + np.random.normal(0, scale = pro)
      state[1] = 1 + np.random.normal(0, scale = pro)
      observation = Fdash[0]*state[0] + Fdash[1]*state[1] + np.random.normal(0, scale = obs)
      states.append(state)
      observations.append(observation)
      return states, observations
   else:
      states, observations = unconst_recursion(G, Fdash, runs-1, pro, obs)
      old_state = states[-1]
      state = [0,0]
      state[0] = G[0]*old_state[0] + G[1]*old_state[1] + np.random.normal(0, scale = pro)
      state[1] = G[2]*old_state[0] + G[3]*old_state[1] + np.random.normal(0, scale = pro)
      observation = Fdash[0]*state[0] + Fdash[1]*state[1] + np.random.normal(0, scale = obs)
      states.append(state)
      observations.append(observation)
      return states, observations
   
def SimCom_elementwise_nonoise(Y,T,level):
# Define a function for solving the NCPO problems with 
# given standard deviations of process noise and observtion noise,
# length of  estimation data and required relaxation level. 

    # Decision Variables
    G = generate_variables("G", n_vars=4, commutative=True)
    Fdash = generate_variables("Fdash", n_vars=2, commutative=True)
    m = generate_variables("m", n_vars=2*(T+1), commutative=True)
    f = generate_variables("f", n_vars=T, commutative=True)

    # Objective
    obj = sum(sum((Y[i,j]-f[j])**2 for j in range(T)) for i in range(Y.shape[0]))

    eqs1 = [f[i] - Fdash[0]*m[2*i+2] - Fdash[1]*m[2*i+3] for i in range(T)]
    eqs2 = [m[i+2] - G[0]*m[i] - G[1]*m[i+1] for i in range(0, 2*T-1, 2)]
    eqs3 = [m[i+2] - G[2]*m[i-1] - G[3]*m[i] for i in range(1, 2*T, 2)]
    eqs = eqs1 + eqs2 + eqs3

    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,Fdash,f,m]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, equalities=eqs) # removeequalities=True doesn't work 
    sdp.solve(solver='mosek')
    #print("sdp: ", sdp)
    #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
    #print(sdp.primal, sdp.dual, sdp.status)
    if (sdp[sum(sum(np.inner(Y[i,j]-f[j], Y[i,j]-f[j]) for j in range(T)) for i in range(Y.shape[0])) ] < 0):
        print("sum((Y[i]-f[i])**2 for i in range(T)) < 0")
        return 
    

    nrmse_sim = 1-sqrt(sdp[sum(np.inner(Y[0,j]-f[j], Y[0,j]-f[j]) for j in range(T)) ])/sqrt(sum(np.inner(Y[0,j]- Y[0].mean(axis=(0)), Y[0,j]- Y[0].mean(axis=(0))) for j in range(T)))
    #nrmse_sim = 1-sqrt(sdp[sum((Y[i]-f[i])**2 for i in range(T))])/sqrt(sum((Y[i]-np.mean(Y))**2 for i in range(T)))

    G_mat = np.array([[sdp[G[0]], sdp[G[1]]], [sdp[G[2]], sdp[G[3]]]])
    Fdash_mat = np.array([[sdp[Fdash[0]], sdp[Fdash[1]]]])

    with np.printoptions(threshold=np.inf):
        print("x_mat: ", sdp.x_mat[0][0])

    with open("monoms.txt","w") as file:
       for i in range(4):
          file.write("G%d %s\n"%(i, str(sdp[G[i]])))
       for i in range(2):
          file.write("Fdash%d %s\n"%(i, str(sdp[Fdash[i]])))
       for i in range(2*T + 2):
           file.write("m%d %s\n"%(i, str(sdp[m[i]])))
       for i in range(T):
           file.write("f%d %s\n"%(i, str(sdp[f[i]])))
       for i in range(T):
           file.write("Fdash*m sum %d %s\n"%(i, str(sdp[Fdash[0]*m[2*i+2]] + sdp[Fdash[1]*m[2*i+3]])))

    for i in range(T):
       print("f",i, " ", sdp[f[i]])
    
    for i in range(2*T + 2):
       print("m",i, " ", sdp[m[i]])

    for i in range(T):
       print("fdas*m ",i, " ", sdp[Fdash[0]*m[2*i+2] + Fdash[1]*m[2*i+3]])

    sdp.write_to_file("sdp_file.csv")
    sdp.save_monomial_index("monomials.txt")

    if(sdp.status != 'infeasible'):
        print(nrmse_sim)
        return nrmse_sim, G_mat, Fdash_mat
    else:
        print('Cannot find feasible solution.')
        return