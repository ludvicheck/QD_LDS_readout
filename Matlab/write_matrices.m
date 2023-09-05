function write_matrices(sys)

sys_name = inputname(1);

writematrix(sys.A,strcat(strcat('data/', sys_name), '_A.csv'))
writematrix(sys.C,strcat(strcat('data/', sys_name), '_C.csv'))
writematrix(sys.K,strcat(strcat('data/', sys_name), '_K.csv'))
writematrix(sys.NoiseVariance,strcat(strcat('data/', sys_name), '_noise.csv'))
