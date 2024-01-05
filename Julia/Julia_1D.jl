using PyCall
using LinearAlgebra
using TSSOS
using DynamicPolynomials
using Random
using Statistics
using DataFrames
using DelimitedFiles
using Dates

py"""
import math
import random
import numpy as np

def generate_C1_data(num_trajectories, T):
   T_1 = 100000 #ns
   mean_C1 = 0.8
   cov_C1 = 0.6
   mean_C0 = 0.1
   cov_C0 = 1

   classes = [0,1]
   trajectories = []

   for _ in range(num_trajectories):
      is_T1 = 1
      trajectory = []
      for point in range(T):
         is_T1 = random.choices(classes, weights = [1 - math.exp(-((point+1) * 8)/ T_1), math.exp(-((point+1) * 8)/ T_1)])#8ns is sampling speed of the device
         if is_T1 == [1]:
            IQ_pt = np.random.normal(mean_C1, cov_C1, size=1)
            trajectory.append(IQ_pt[0])
                
         else:
            remaining = int(T - point)
            IQ_pts = np.random.normal(mean_C0, cov_C0, size=remaining)
            for j in IQ_pts[:]:
                trajectory.append(j)
            break
         
      trajectories = list(trajectories)
      trajectories.append(trajectory)
      trajectories = np.array(trajectories)

   return trajectories
"""

function observation(num_trajectories,T,trajectories)
    return copy(trajectories[1:num_trajectories, 1:T])
end

function parameter_estimation1(Y, order)
    # training process
    
    data_dims = size(Y)
    num_trajectories = data_dims[1]
    T=data_dims[2]
    
    @polyvar G[1:4] Fdash[1:2] p[1:T] q[1:2*T] f[1:T] m[1:2*(T+1)];
    var=vcat(G,Fdash,m,f,p,q);

    # constraints
    ine1 = [f[i] - Fdash[1]*m[2*i+1] - Fdash[2]*m[2*i+2] - p[i] for i in 1:T];
    ine2 = [-f[i] + Fdash[1]*m[2*i+1] + Fdash[2]*m[2*i+2] + p[i] for i in 1:T];
    ine3 = [m[i+2] - G[1]*m[i] - G[2]*m[i+1] - q[i]  for i in 1:2:2*T];
    ine4 = [-m[i+2] + G[1]*m[i] + G[2]*m[i+1] + q[i]  for i in 1:2:2*T];
    ine5 = [m[i+2] - G[3]*m[i-1] - G[4]*m[i] - q[i]  for i in 2:2:2*T];
    ine6 = [-m[i+2] + G[3]*m[i-1] + G[4]*m[i] + q[i]  for i in 2:2:2*T];
    #ine7 = [m[1] - 1]#1/sqrt(2)
    #ine8 = [-m[1] + 1]
    #ine9 = [m[2] - 1]
    #ine10 = [-m[2] + 1]
    


    #objective
    obj=sum( (Y[i,j]-f[j])^2 for j=1:T, i=1:num_trajectories) +
            sqrt(num_trajectories)*0.005*sum(p[i]^2 for i in 1:T) + 
            sqrt(num_trajectories)*0.001*sum(q[i]^2 for i in 1:2*T)
    
    #println(obj)

    # pop
    pop=vcat(obj,ine1,ine2,ine3,ine4,ine5,ine6);

    # solve model
    opt,sol,data=tssos_first(pop,var,order,TS="MD",solution=true);
    
    #println("sol: ", sol)
    #println("opt: ", opt)
    #println("data: ", data)
    return opt, sol, data 
end

#trajectories = readdlm("QD_1D_trajectories.txt", ' ')
trajectories = readdlm("tohoku_state0_trajectories.csv", ',', Float64)
trajectories = trajectories[2:end, :]

num_trajectories=[1,1000] 

current_datetime = Dates.now()
datetime_str = Dates.format(current_datetime, "yyyy-mm-dd_HH-MM-SS")
file_name = "tohoku_state0_free_init_$datetime_str.txt"
script_dir = @__DIR__
file_path = joinpath(script_dir, file_name)

for T in 22:50
    for num_tr in num_trajectories
        for order in 1:1
            println("T=",T,", number of trajectories=", num_tr, ", order=", order)
            Y=observation(num_tr,T,trajectories)
            opt, sol, data = parameter_estimation1(Y, order)
            if data.flag == 0
                status = "Global optimum!"
            else
                status = "Local solver failed or obtained local optimum with Ipopt."
            end

            G = sol[1:4]
            Fdash = sol[5:6]
            ms = sol[7:2*(T+1)+6]
            fs = sol[2*(T+1)+7:2*(T+1)+T+6]
            ps = sol[2*(T+1)+T+7:2*(T+1)+T+2*T+6]
            os = sol[*(T+1)+T+2*T+7: 2*(T+1)+T+2*T+T+6]

            open(file_path, "a") do file
                write(file, "T = $T, number of trajectories = $num_tr, order = $order", "\n")
                write(file, "status = $status", "\n")
                write(file, "optimum = $opt", "\n")
                write(file, "G = $G", "\n")
                write(file, "Fdash = $Fdash", "\n")
                write(file, "ms = $ms", "\n")
                write(file, "fs = $fs", "\n")
                write(file, "process noise = $ps", "\n")
                write(file, "observation noise = $os", "\n")
                write(file, " ", "\n")
            end
        end
    end
end