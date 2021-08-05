% A simple script to show that local is global
% min_{G Q_i} sum_i||X_i*Q_i-G||
% st G^tG = I;
% X_i is tall and full rank



addpath /scratch/sagar/Projects/matlab/lib/tensor_toolbox
addpath ../PAMI_sim/functions
addpath ../PAMI_sim/algos
addpath ../PAMI_sim/cg_matlab

clear;
clc;
close;

TotalTrial = 1;
dev = .1;
n_bits = 3;
MaxIt = 2500; 
InnerIt = 10;

for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])

    I = 5;
    K = 10;
    latent_dim = 10;
    M = 80;
    L = 1000;
    m = 5;
    prob_observed = 0.3;
    latent_rep = randn(latent_dim, L);
    for i=1:I
        A{i} = randn(M, latent_dim);
        F{i} = diag(rand(1,latent_dim)>prob_observed);
        X{i} = (A{i}*F{i}*latent_rep)' + 0.1*randn(L,M);
    end

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


    % G_ini = randn(L,K);
    % for i=1:I
    %     Q_ini{i} = randn(M,K);
    % end

    [Q2,G_2,obj2(trial,:),~,St2, t2] = LargeGCCA_federated_stochastic( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', true, 'federated', false, 'compress_g', false);

    [Q1,G_1,obj1(trial,:),~,St1, t1] = LargeGCCA_federated_stochastic( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', true, 'federated', true, 'compress_g', true);

    time_proposed_1(trial) = toc;
   
end   
save('data/compress_g.mat', 'obj1', 'obj2')

% save('data/rand_compressor_2bits_2norm.mat', 'obj2' )

%%

plot(obj1, 'LineWidth', 2); hold on;
plot(obj2, 'LineWidth', 2); hold on;
% plot(obj3, 'LineWidth', 2); hold on;
legend('Full resolution', 'Federated');
title('Wireless sensor network, 10 views, latent_dim=10, K=10, (L,M) = (1000, 800)')
