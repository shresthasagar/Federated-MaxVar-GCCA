% A simple script to show that local is global
% min_{G Q_i} sum_i||X_i*Q_i-G||
% st G^tG = I;
% X_i is tall and full rank



addpath /scratch/sagar/Projects/matlab/lib/tensor_toolbox
addpath ../fu2017/functions
addpath ../fu2017/algos
addpath ../fu2017/cg_matlab


clear;
clc;
close;

 
 
TotalTrial = 1;
dev = .1;
n_bits = 4;
MaxIt = 300; 
InnerIt = 10;
for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])
    % I = 3;
    % L = 100000;
    % M = 50000; 
    % N = 100;
    % K = 10;
    % m = 100;
    % sparsity_level = 1e-6;
    
    % Z = sprandn(L,N,sparsity_level);
    % for i=1:I
    %     A{i}=sprandn(N,M,.00001);
    %     X{i}=Z*A{i};
    %     X{i}=sparse(X{i});
    % end

    I = 3;
    L = 50000;
    M = 30000;
    N = 30000;
    K = 10;
    m = 100;
    Z = sprandn(L,N, 1e-4);%*diag(2+1*randn(N,1));
    for i=1:I
        A{i}=sprandn(N,M, 0.00001);%+ 1*eye(N,N);
%         [Ua,~,Va]=svd(A{i});
%         A{i}=Va(1:N,:)+ dev*randn(N,M);
        
        X{i}=Z*A{i}; % + .1*randn(L,M); 
        % condnum(i)=cond((1/L)*X{1}'*X{1});
    end
    % Zf = full(Z);
    % [Uz, ~, ~]  = svd(Zf, 0);
    % Ubeta = Uz(:, K+1:end);


    %% How to initialize is another problem...
     
    r = 0.1;
    tic;
    filename = ['data/rand_',num2str(trial)];
    % [ G_ini,Q_ini,Ux,Us,UB,cost_MLSA(trial),Li ] = MLSA( X,K,m,r);
    % % timeMLSA(trial) =toc;
    % cost_MLSA(trial)
    % save(filename,'X','G_ini','Q_ini','cost_MLSA','Li');
%     
       

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
    % [Q3,G_3,obj3(trial,:),~,St2, t2] = LargeGCCA_federated_stochastic( X, K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', true, 'federated', false, 'compress_g', false);
    [Q2,G_2,obj2(trial,:),~,St1, t1] = LargeGCCA_federated_stochastic( X, K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', true, 'federated', true, 'compress_avg', true);

    [Q1,G_1,obj1(trial,:),~,St1, t1] = LargeGCCA_federated_stochastic( X, K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', true, 'federated', true, 'compress_g', true);
    
    

    % [Q2,G_2,obj2(trial,:),~,St2, t2] = LargeGCCA_federated_stochastic( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', true, 'federated', true);

    % [Q3,G_3,obj3(trial,:),~,St3, t3] = LargeGCCA_federated_stochastic( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', n_bits, 'sgd', false, 'batch_size', 10000, 'rand_compress', true, 'federated', false);

    % [Q,G_2,obj2(trial,:),~,St2, t2] = LargeGCCA_federated_stochastic( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1, 'Reg_type', 'fro', 'nbits', 2, 'sgd', false, 'batch_size', 1000);

    % [Q2,G_2,obj2(trial,:),dist2,St2] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none');

    time_proposed_1(trial) = toc;
   
end   
save('data/compress_g.mat', 'obj1', 'obj2')

loglog(obj1, 'LineWidth', 2); hold on;
loglog(obj2, 'LineWidth', 2); hold on;
% plot(obj3, 'LineWidth', 2); hold on;
legend('with G compression', 'with avg compression');
title('full resolution and compressed G with random compressor 3 bits, 1000 inner iter')
