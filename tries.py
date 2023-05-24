import sys
sys.path.append("\anaconda3\pkgs") 
from inputlds import*
from QD_LDS_functions import*
from ncpol2sdpa import*
import numpy as np
import pandas as pd
from math import sqrt

obs =0.2
pro = 0.1
level = 1
measurements = 1
T = 20

"""

#try with Hazan system but Kalman without learned matrices
g = np.matrix([[0.9,0.2],[0.1,0.1]])
f_dash = np.matrix([[1.0,1.0]])

if g.shape[0] == 1:
    pro_mat = pro
else:
    pro_mat = pro*np.eye(g.shape[0])

if f_dash.shape[0] == 1:
    obs_mat = obs
else:
    obs_mat = obs*np.eye(f_dash.shape[0])


Y= []
for i in range(measurements):
    Y.append(data_generation(g,f_dash,pro,obs,T))
Y = np.array(Y)

nmrse, sdp = multiple_SimCom(Y ,T, level)
print("nrmse: ", nmrse)

preds, errors = Kalman_filter(g, f_dash.T, pro_mat, obs_mat, Y[0])
print("preds: ", preds )
print("errors: ", errors)





"""
#try with simulated IQ data and Kalman filter with system matrices that seem to me "reasonable"
G = np.eye(2)
Fdash = np.array([[0.1, 0.8], [0.2, 0.1]])
obs2D = [[0.2, 0], [0, 0.2]]
cov_C1 = [[0.6, 0], [0, 0.6]]

trajectories = generate_C1_data(measurements,T)

nmrse, sdp = multiple_SimCom(trajectories ,T, level)
print("nrmse: ", nmrse)
print("prim: ", sdp.primal, "dual: ", sdp.dual, "stat: ", sdp.status, "nmrse:", nmrse)

preds2D, errors2D = Kalman_filter(G, Fdash.T, obs2D, cov_C1, trajectories[0])
print("preds: ", preds2D )
print("errors: ", errors2D)


#It is possible also to try SimCom_elementwise function (instead of multiple_SimCom), which approaches the NCPOP problem with variables as matrix elements.
#This is probably bad approach, but for Hazan system it still provides good results