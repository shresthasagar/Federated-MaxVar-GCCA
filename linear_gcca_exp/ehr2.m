addpath /scratch/sagar/Projects/matlab/lib/tensor_toolbox
addpath ../PAMI_sim/functions
addpath ../PAMI_sim/algos
addpath ../PAMI_sim/cg_matlab

clear;
clc;
close;

I = 3

% Load data
view1_filename = '/scratch/sagar/Projects/federated_max_var_gcca/ehr/data/cms_view1_matlab.mat';
view2_filename = '/scratch/sagar/Projects/federated_max_var_gcca/ehr/data/cms_view2_matlab.mat';
view3_filename = '/scratch/sagar/Projects/federated_max_var_gcca/ehr/data/cms_view3_matlab.mat';

% combined views data
view_filename = '/scratch/sagar/Projects/federated_max_var_gcca/ehr/data/combined_views_matlab.mat';
view = load(view_filename);


TotalTrial = 1;
dev = .1;
n_bits = 2;
MaxIt = 1500; 
InnerIt = 10;
for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])

    K = 5;
    m = 20;
    rand_perm = randperm(1000);
    X{1} = double(view.x1);
    X{2} = double(view.x2);
    X{3} = double(view.x3);

    % mean center
    X{1} = X{1} - mean(X{1});
    X{2} = X{2} - mean(X{2});
    X{3} = X{3} - mean(X{3});

    L = size(X{1}(1));
    M = size(X{1}(2));
    
    %% How to initialize is another problem...
    r = 0.1;
    tic;
    filename = ['data/rand_',num2str(trial)];
    [ G_ini,Q_ini,Ux,Us,UB,cost_MLSA(trial),Li ] = MLSA( X,K,m,r);
    % timeMLSA(trial) =toc;
    cost_MLSA(trial)
    save(filename,'X','G_ini','Q_ini','cost_MLSA','Li');
    
    tic
    % load from file
    % filename = ['data/rand_',num2str(trial)]; 
    init_vars  = load(filename);

    X = init_vars.X;
    G_ini = init_vars.G_ini;
    Q_ini = init_vars.Q_ini;
    cost_MLSA = init_vars.cost_MLSA;
    Li = init_vars.Li;

    %%
    % XX = zeros(L,M);
    % for i=1:i
    %     XX = XX + X{i}*inv(X{i}'*X{i})*X{i}';
    % end
    % [Um, ~,~] = svd(M,0);

    % [Q1,G_1,obj1(trial,:),~,St1, t1] = LargeGCCA_federated_stochastic( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', false, 'federated', true);

    % [Q2,G_2,obj2(trial,:),~,St2, t2] = LargeGCCA_federated_stochastic( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', false, 'federated', false);

    [Q3,G_3,obj3(trial,:),~,St3, t3] = LargeGCCA_federated_stochastic( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', true, 'federated', false);

    % [Q,G_2,obj2(trial,:),~,St2, t2] = LargeGCCA_federated_stochastic( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1, 'Reg_type', 'fro', 'nbits', 2, 'sgd', false, 'batch_size', 1000);

    % [Q2,G_2,obj2(trial,:),dist2,St2] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none');

    time_proposed_1(trial) = toc;
   
end   
save('data/results/ehr_G1.mat', 'G_3');