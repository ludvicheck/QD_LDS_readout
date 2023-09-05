function [def,N4sid,ssa]=mult_sys_idents(T,Y,inputs, nx)

Ts = 5;  % 5 nanoseconds
steps = size(Y,1);
time_vector = (0:steps-1) * Ts;
N = size(Y,2)/2;%number of trajectories

sampling_instants_cell = cell(1, N);

for i = 1:N
    sampling_instants_cell{i} = time_vector;
end

z = iddata(Y(:,1:2),inputs,T);

if N >= 2
    for i = 2:size(Y,2)/2
        traj_iddata = iddata(Y(:,2*i-1:2*i),inputs,T);
        z = merge(z, traj_iddata);
    end
end

z.OutputName = {'Amplitude', 'Phase'};
z.TimeUnit = 'nanoseconds';
%z.Ts = 5;
z.SamplingInstants = sampling_instants_cell;

opt = ssestOptions;
opt.Focus = 'simulation';
%opt.InitialState = [1,0];
%opt.EnforceStability = true;
def=ssest(z,nx,opt); 

optss=n4sidOptions;
optss.Focus = 'simulation';
optss.EnforceStability = true;
N4sid=n4sid(z,nx,optss);

optssa=optss;
optssa.N4Weight='SSARX';
ssa = n4sid(z,nx,optssa);

compare(z,def,N4sid,ssa)