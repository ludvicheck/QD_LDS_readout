import sys
sys.path.append("\anaconda3\pkgs") 
from inputlds import*
from QD_LDS_functions import*
from ncpol2sdpa import*
import numpy as np
from math import sqrt
import random


level = 1
measurements = 10
T = 20

#generate training trajectories - later will be obtained from real measurement
trajectories_C0 = generate_C1_data(measurements,T)
trajectories_C1 = generate_C0_data(measurements,T)

#NCPOP learning of the system matrices for both classes
nrmse0, sdp0 = multiple_SimCom(trajectories_C0 ,T, level)
nrmse1, sdp1 = multiple_SimCom(trajectories_C1 ,T, level)

#generate new trajectory of apriory unknown class - later will be obtained from real measurement
classes = [0,1]
class_label = random.choices(classes, weights = [1/2, 1/2])
if class_label == [0]:
    print("generated as class: 0")
    new_trajectory = generate_C0_data(1,T)
else:
    print("generated as class: 1")
    new_trajectory = generate_C1_data(1,T)

#preds0, errors0 = Kalman_filter(g0, f_dash0.T, pro_mat0, obs_mat0, new_trajectory)
#preds1, errors1 = Kalman_filter(g1, f_dash1.T, pro_mat1, obs_mat1, new_trajectory)

#decission = labelled_decission(errors0, errors1)
#print("assigned class:", decission)
