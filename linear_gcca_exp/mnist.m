addpath ../algos
addpath ../cg_matlab

clear;
clc;
close;
data = double(load('data/mnist_train.mat').X);

[I L M] = size(data);
for i=1:I 
    X{i} = squeeze(data(i,:,:));
end
K = 10;
m = 100;
r = 0.01;

tic;
filename = ['data/mnist_initvals.mat'];

% [ G_ini,Q_ini,Ux,Us,UB,cost_MLSA,Li ] = MLSA( X,K,m,r);

% save(filename,'X','G_ini','Q_ini','cost_MLSA','Li');

MaxIt = 1000; 
toc;

% load from file
init_vars  = load(filename);

X = init_vars.X;
G_ini = init_vars.G_ini;
Q_ini = init_vars.Q_ini;
cost_MLSA = init_vars.cost_MLSA;
Li = init_vars.Li;

%%
% [Q,G_1,obj1,~,St1, time1] = LargeGCCA_federated_stochastic(X, K, 'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1, 'Reg_type', 'none', 'sgd', false, 'batch_size', 10000);
% save('data/real_data_3view_fullgd_obj1.mat', 'obj1', 'time1');

% [Q,G_2,obj2,~,St2, time2] = LargeGCCA_federated_stochastic(X, K, 'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1, 'Reg_type', 'none', 'sgd', false, 'batch_size', 10000);
% save('data/real_data_3view_sgd_obj1.mat', 'obj2', 'time2');

[Q,G,obj2,dist2,St2] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1, 'Reg_type', 'none');

save('data/results/mnist_gcca.mat', 'obj2', 'Q', 'G');