import sys
from inputlds import*
from QD_LDS_functions import*
from ncpol2sdpa import*
import numpy as np
import pandas as pd
from math import sqrt

num_trajectories = 50
timesteps = 200

'''
G_mat = pd.read_csv('data/AN4.csv', sep=',', header=None)
G_mat = np.array(G_mat.values)
Fdash_mat = pd.read_csv('data/CN4.csv', sep=',', header=None)
Fdash_mat = np.array(Fdash_mat.values)
Y = pd.read_csv('data/Y30_validate.csv', sep=',', header=None)
Y = np.array(Y.values)
print(G_mat)

obs =0.02 #0.2
pro = 0.01 #0.1

if G_mat.shape[0] == 1:
    pro_mat = pro
else:
    pro_mat = pro*np.eye(G_mat.shape[0])

if Fdash_mat.shape[0] == 1:
    obs_mat = obs
else:
    obs_mat = obs*np.eye(Fdash_mat.shape[0])

preds, errors = Kalman_filter(G_mat, Fdash_mat.T, pro_mat, obs_mat, Y)
print("preds: ", preds )
print("errors: ", errors)
'''

Y = generate_C0_data(num_trajectories, timesteps)
print(Y)
Y = np.transpose(Y, (1,0,2))
Y = np.reshape(Y, (timesteps, 2*num_trajectories))
print(Y)

Ydf=pd.DataFrame(Y)
Ydf.to_csv('data/YC0_50_200.csv',index=False)