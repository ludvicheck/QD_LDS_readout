nx=2;
num_trajectories = 1;
T=200;

outputs = YC050200(1:T,1:2*num_trajectories);
inputs = zeros(T,1);

[defC0_1_200,N4sidC0_1_200,ssaC0_1_200]=mult_sys_idents(T,outputs, inputs, nx);

%writematrix(N4sid30_mult.A,'AN4_mult.csv')
%writematrix(N4sid30_mult.C,'CN4_mult.csv')
%writematrix(Y30validate,'Y30_validate.csv')