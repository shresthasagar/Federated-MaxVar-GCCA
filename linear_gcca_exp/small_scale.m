% A simple script to show that local is global
% min_{G Q_i} sum_i||X_i*Q_i-G||
% st G^tG = I;
% X_i is tall and full rank

addpath /scratch/sagar/Projects/matlab/lib/tensor_toolbox
% addpath ../fu2017/functions
% addpath ../fu2017/algos
% addpath ../fu2017/cg_matlab
addpath src

clear;
clc;
close;

% Use the simulation filename in the format "scale_counter"
simulation_filename = 'small_1.mat'

% Commonly available data inside the simulation file
%   1. Objective value at each iteration
%   2. Elapsed time at each iteration
%   3. Data generating conditions (L, M, N, K, m, mu)

TotalTrial = 1;
mu = .1;
noise_var = 0.1;
MaxIt = 3000;
InnerIt = 10;
Nbits = 3;
r = .01;

for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])
    % dense: sanity check
    I = 3;
    L = 500;
    Lbad = 0;
    M = 25;
    N = 20;
    K = 5;
    m = 10;
    Z = randn(L,N);
    for i=1:I
        A{i} = randn(N,M);
        X{i} = Z*A{i} + noise_var*randn(L,M); 
        condnum(i) = cond((1/L)*X{1}'*X{1});
    end
    
    cond_mean(trial)=mean( condnum(i));
    
    %% computing the global solution
    tic
    ZZ = zeros(L,L);
    MM = zeros(L,L); M_clean=zeros(L,L);
    for i=1:I
        % M_clean = M_clean + X_clean{i}*((X_clean{i}'*X_clean{i}+r*eye(M))\X_clean{i}')
        MM = MM + L*X{i}*(((1/L)*X{i}'*X{i}+r*eye(M))\X{i}');
        [Ux{i},sigx{i},~] = svd(X{i});
        ZZ = ZZ+Ux{i}(:,1:M)*inv(sigx{i}(1:M,:).^2+r*eye(M))*Ux{i}(:,1:M)';  % interesting question: why is this important?
    end
    [Um,Sm,Vm] = svd(MM,0); 
    DiagSm = Sm/I;  
    
    Ubeta = Um(:,K+1:end);
    [valMM,order_ind] = sort(diag(Sm),'descend');
    G = Um(:,order_ind(1:K));%*Vm(:,1:K)';
     
    cost_global = 0;
    for i = 1:I 
        Q{i} = (1/sqrt(L))*(((1/L)*X{i}'*X{i}+r*eye(M))\(X{i}'*G));
        cost_global= (1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+cost_global+(r/2)*sum(sum(Q{i}.^2));
    end
    cost_optimal(trial)=cost_global;
    time_global = toc
    
    %% MVLSA 
    tic;
    [ G_ini,Q_ini,Ux,Us,UB,cost_mlsa(trial),Li ] = MLSA( X,K,m,r);
    % dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
    time_mlsa =toc;
    
    %% Full resolution MAX-VAR GCCA with warm start
    disp(['Full resolution with warm start ...'])
    [Q1,G_1,obj_warm_full_res(trial,:),~,St1, t_warm_full_res] = LargeGCCA_distributed_stochastic( X,  K, ...
                                                                        'G_ini',G_ini, ...
                                                                        'Q_ini',Q_ini, ...
                                                                        'r',r, ...
                                                                        'algo_type','plain', ...
                                                                        'Li',Li, ...
                                                                        'MaxIt',MaxIt, ...
                                                                        'Inner_it',InnerIt, ...
                                                                        'Reg_type', 'fro',  ...
                                                                        'distributed', false,  ...
                                                                        'nbits', Nbits,  ...
                                                                        'sgd', false,  ...
                                                                        'batch_size', 10000,  ...
                                                                        'rand_compress', false,  ...
                                                                        'compress_g', false, ...
                                                                        'print_log', true); 
                                                                        
    % Ditributed MAX-VAR GCCA with warm start
    disp(['Distributed with warm start ...'])
    [Q1,G_1,obj_warm_distr(trial,:),~,St1, t_warm_distr] = LargeGCCA_distributed_stochastic( X,  K, ...
                                                                        'G_ini',G_ini, ...
                                                                        'Q_ini',Q_ini, ...
                                                                        'r',r, ...
                                                                        'algo_type','plain', ...
                                                                        'Li',Li, ...
                                                                        'MaxIt',MaxIt, ...
                                                                        'Inner_it',InnerIt, ...
                                                                        'Reg_type', 'fro',  ...
                                                                        'distributed', true,  ...
                                                                        'nbits', Nbits,  ...
                                                                        'sgd', false,  ...
                                                                        'batch_size', 10000,  ...
                                                                        'rand_compress', true,  ...
                                                                        'compress_g', true, ...
                                                                        'print_log', true); 
   
    % Randomly initialize the G and Q
    G_ini = randn(L,K);
    for i=1:I
        Q_ini{i} = randn(M,K);
    end

    % Full resolution MAX-VAR GCCA with randn start
    disp(['Full resolution with randn start ...'])
    [Q1,G_1,obj_randn_full_res(trial,:),~,St1, t_randn_full_res] = LargeGCCA_distributed_stochastic( X,  K, ...
                                                                        'G_ini',G_ini, ...
                                                                        'Q_ini',Q_ini, ...
                                                                        'r',r, ...
                                                                        'algo_type','plain', ...
                                                                        'Li',Li, ...
                                                                        'MaxIt',MaxIt, ...
                                                                        'Inner_it',InnerIt, ...
                                                                        'Reg_type', 'fro',  ...
                                                                        'distributed', false,  ...
                                                                        'nbits', Nbits,  ...
                                                                        'sgd', false,  ...
                                                                        'batch_size', 10000,  ...
                                                                        'rand_compress', false,  ...
                                                                        'compress_g', false, ...
                                                                        'print_log', true); 

    % Ditributed MAX-VAR GCCA with randn start
    disp(['Distributed with randn start ...'])
    [Q1,G_1,obj_randn_distr(trial,:),~,St1, t_randn_distr] = LargeGCCA_distributed_stochastic( X,  K, ...
                                                                        'G_ini',G_ini, ...
                                                                        'Q_ini',Q_ini, ...
                                                                        'r',r, ...
                                                                        'algo_type','plain', ...
                                                                        'Li',Li, ...
                                                                        'MaxIt',MaxIt, ...
                                                                        'Inner_it',InnerIt, ...
                                                                        'Reg_type', 'fro',  ...
                                                                        'distributed', true,  ...
                                                                        'nbits', Nbits,  ...
                                                                        'sgd', false,  ...
                                                                        'batch_size', 10000,  ...
                                                                        'rand_compress', true,  ...
                                                                        'compress_g', true); 
    
end  


%% Visualization of cost vs iteration

figure(1)

max_iter = max([length(obj_warm_full_res), length(obj_warm_distr), length(obj_randn_full_res), length(obj_randn_distr)]);

% Optimal
h1 = loglog([1:max_iter], cost_optimal*ones(1, max_iter), '-k', 'linewidth', 2); hold on;

% MVLSA 
h2 = loglog([1:max_iter], cost_mlsa*ones(1, max_iter), '-g', 'linewidth', 2); hold on;

% Full Resolution warm start
h3 = loglog([1:length(obj_warm_full_res)], obj_warm_full_res, '-r', 'linewidth', 2); hold on

% Distributed warm start
h4 = loglog([1:length(obj_warm_distr)], obj_warm_distr, '--b', 'linewidth', 2); hold on

% Distributed randn start
h5 = loglog([1:length(obj_randn_full_res)], obj_randn_full_res, '-m', 'linewidth', 2); hold on

% Distributed randn start
h6 = loglog([1:length(obj_randn_distr)], obj_randn_distr, '--c', 'linewidth', 2); hold off

legend([h1,h2,h3,h4,h5,h6], {'Optimal', 'MVLSA', 'AltMaxVar (warm start)', 'Ditributed AltMaxVar (warm start)', 'AltMaxVar (randn start)', 'Distributed AltMaxVar (randn start)'});
% legend('Optimal', 'MVLSA', 'AltMaxVar (warm start)', 'Ditributed AltMaxVar (warm start)', 'AltMaxVar (randn start)', 'Distributed AltMaxVar (randn start)');
% set(gca,'fontsize',14)
xlabel('iterations','fontsize',14)
ylabel('cost value','fontsize',14)
% print('-depsc','lambda_1')
xlim([1 max_iter])

set(gcf, 'PaperPosition', [0 0 7 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [7 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'data/simulation_results/small_1', 'pdf') %Save figure

%% Visualization of Communication cost (bytes) vs Iteration
initial_comm_cost = 32;

per_iter_distr = Nbits;
per_iter_full_res = 32;

comm_cost_iter_distr(1) = per_iter_distr;
for iter=2:max_iter
    comm_cost_iter_distr(i) = comm_cost_iter_idstr(i-1) + per_iter_distr;
end

comm_cost_iter_full_res(1) = per_iter_full_res;
for iter=2:max_iter
    comm_cost_iter_full_res(i) = comm_cost_iter_full_ress(i-1) + per_iter_full_res;
end

figure(2)
h1 = loglog(comm_cost_iter_full_res, obj_warm_full_res)
h2 = loglog(comm_cost_iter_distr, obj_warm_distr)

legend([h1,h2], {'AltMaxVar (warm start)', 'Distributed AltMaxVar (warm start)'})

