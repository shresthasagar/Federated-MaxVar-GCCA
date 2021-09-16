% A simple script to show that local is global
% min_{G Q_i} sum_i||X_i*Q_i-G||
% st G^tG = I;
% X_i is tall and full rank



addpath ../fu2017/algos
addpath ../fu2017/cg_matlab

clear;
clc;
close;

data = load('../real_data/Europarl_all.mat');
[~, I] = size(data.data);
I = 3;

for i=1:I 
    X{i} = data.data{i};
end

[L, M] = size(X{1});
N = 50000;
K = 10;
m = 100;
r = 0.1;

%% How to initialize is another problem...
    
tic;
filename = ['data/real_data_mvlsa_3view.mat'];

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
[Q,G_1,obj1,~,St1, time1] = LargeGCCA_distributed_stochastic(X, K, 'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1, 'Reg_type', 'none', 'sgd', true, 'batch_size', 10000, 'distributed', true, 'rand_compress', true, 'compress_g', true);
save('data/real_data_3view_fullgd_obj1.mat', 'obj1', 'time1');

% [Q,G_2,obj2,~,St2, time2] = LargeGCCA_federated_stochastic(X, K, 'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1, 'Reg_type', 'none', 'sgd', false, 'batch_size', 10000);
% save('data/real_data_3view_sgd_obj1.mat', 'obj2', 'time2');

% [Q2,G_2,obj2,dist2,St2] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1, 'Reg_type', 'none');

% save('data/real_data2_obj2.mat', 'obj2');