%---------------------------------------------------
% Communication cost (bits per variable) vs batch size
% Wall-clock time (seconds) vs batch size
%---------------------------------------------------


addpath /scratch/sagar/Projects/matlab/lib/tensor_toolbox
addpath ../src

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
MaxIt = 50;
InnerIt = 10;
Nbits = 3;
r = .01;

batch_size = []
inner_iters = [10]

for inner_it_idx=1:length(inner_iters)
    InnerIt = inner_iters(inner_it_idx)
    if InnerIt== 100
        MaxIt = 100;
    end
    for trial = 1:TotalTrial
        disp(['at trial ',num2str(trial)])
        % dense: sanity check
        I = 3;
        L = 50000;
        M = 5000;
        N = 30000;
        K = 10;
        m = 100;
        Z = sprandn(L, N, 1e-4);%*diag(2+1*randn(N,1));
        for i=1:I
            A{i}=sprandn(N,M, 0.0001);            
            X{i}=Z*A{i};
        end
    
        %% computing the global solution
        % tic
        % ZZ = zeros(L,L);
        % MM = zeros(L,L); M_clean=zeros(L,L);
        % for i=1:I
        %     % M_clean = M_clean + X_clean{i}*((X_clean{i}'*X_clean{i}+r*eye(M))\X_clean{i}')
        %     MM = MM + L*X{i}*(((1/L)*X{i}'*X{i}+r*eye(M))\X{i}');
        %     [Ux{i},sigx{i},~] = svd(X{i});
        %     ZZ = ZZ+Ux{i}(:,1:M)*inv(sigx{i}(1:M,:).^2+r*eye(M))*Ux{i}(:,1:M)';  % interesting question: why is this important?
        % end
        % [Um,Sm,Vm] = svd(MM,0); 
        % DiagSm = Sm/I;  
        
        % Ubeta = Um(:,K+1:end);
        % [valMM,order_ind] = sort(diag(Sm),'descend');
        % G = Um(:,order_ind(1:K));%*Vm(:,1:K)';
        
        % cost_global = 0;
        % for i = 1:I 
        %     Q{i} = (1/sqrt(L))*(((1/L)*X{i}'*X{i}+r*eye(M))\(X{i}'*G));
        %     cost_global= (1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+cost_global+(r/2)*sum(sum(Q{i}.^2));
        % end
        % cost_optimal(trial)=cost_global;
        % time_global = toc
        
        %% MVLSA 
        tic;
        filename = '../data/rand_1.mat';
        % [ G_ini,Q_ini,Ux,Us,UB,cost_mlsa(trial),Li ] = MLSA( X,K,m,r);
        % % dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
        % time_mlsa =toc;

        init_vars  = load(filename);
        X = init_vars.X;
        G_ini = init_vars.G_ini;
        Q_ini = init_vars.Q_ini;
        cost_MLSA = init_vars.cost_MLSA;
        Li = init_vars.Li;

        % If we use frobenius norm regularization the loss will stay quite high
        
        %% Full resolution MAX-VAR GCCA with warm start

        % [Q2,G_2,obj2(trial,:),~,St1, t1] = LargeGCCA_distributed_stochastic( X, K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',InnerIt, 'Reg_type', 'none', 'nbits', Nbits, 'sgd', false, 'batch_size', 10000, 'rand_compress', true, 'distributed', true, 'compress_g', true, 'print_log', true);

        disp(['Full resolution with warm start ...'])
        [Q1,G_1,obj_warm_full_res(trial,inner_it_idx,:),~,St1, t_warm_full_res] = LargeGCCA_distributed_stochastic( X,  K, ...
                                                                            'G_ini',G_ini, ...
                                                                            'Q_ini',Q_ini, ...
                                                                            'r',r, ...
                                                                            'algo_type','plain', ...
                                                                            'Li',Li, ...
                                                                            'MaxIt',MaxIt, ...
                                                                            'Inner_it',InnerIt, ...
                                                                            'Reg_type', 'none',  ...
                                                                            'distributed', false,  ...
                                                                            'nbits', Nbits,  ...
                                                                            'sgd', false,  ...
                                                                            'batch_size', 1000,  ...
                                                                            'rand_compress', false,  ...
                                                                            'compress_g', false, ...
                                                                            'print_log', true); 
                                                                            
        % Ditributed MAX-VAR GCCA with warm start
        disp(['Distributed with warm start ...'])
        [Q1,G_1,obj_warm_distr(trial,inner_it_idx,:),~,St1, t_warm_distr] = LargeGCCA_distributed_stochastic( X,  K, ...
                                                                            'G_ini',G_ini, ...
                                                                            'Q_ini',Q_ini, ...
                                                                            'r',r, ...
                                                                            'algo_type','plain', ...
                                                                            'Li',Li, ...
                                                                            'MaxIt',MaxIt, ...
                                                                            'Inner_it',InnerIt, ...
                                                                            'Reg_type', 'none',  ...
                                                                            'distributed', true,  ...
                                                                            'nbits', Nbits,  ...
                                                                            'sgd', false,  ...
                                                                            'batch_size', 10000,  ...
                                                                            'rand_compress', true,  ...
                                                                            'compress_g', true, ...
                                                                            'print_log', true);     
    end  
end

%% Visualization of Communication cost (bytes) vs Iteration
figure(1)



% Full Resolution warm start
h1 = loglog([1:length(squeeze(obj_warm_full_res(1,1,:)))], squeeze(obj_warm_full_res(1,1,:)), '-r', 'linewidth', 2); hold on

% Distributed warm start
h2 = loglog([1:length\squeez((obj_warm_distr(1,1,:)))], squeeze(obj_warm_distr(1,1,:)), '--b', 'linewidth', 2); hold on

% Full res 
h3 = loglog([1:length(squeeze(obj_warm_full_res(1,2,:)))], squeeze(obj_warm_full_res(1,2,:)), '-m', 'linewidth', 2); hold on

% Distributed 
h4 = loglog([1:length(squeeze(obj_warm_distr(1,2,:)))], squeeze(obj_warm_distr(1,2,:)), '--c', 'linewidth', 2); hold off


% % Full res 
% h3 = loglog([1:length(obj_warm_full_res(1,3,:))], obj_warm_full_res(1,3,:), '-m', 'linewidth', 2); hold on

% % Distributed 
% h4 = loglog([1:length(obj_warm_distr(1,3,:))], obj_warm_distr(1,3,:), '--c', 'linewidth', 2); hold off


% legend([h1,h2,h3,h4,h5,h6], {'AltMaxVar (T=1)', 'Distributed AltMaxVar (T=1)','AltMaxVar (T=10)', 'Distributed AltMaxVar (T=10)','AltMaxVar (T=100)', 'Distributed AltMaxVar (T=100)'})
legend([h1,h2,h3,h4], {'AltMaxVar (T=1)', 'Distributed AltMaxVar (T=1)','AltMaxVar (T=10)', 'Distributed AltMaxVar (T=10)'})

set(gca,'fontsize',14)
xlabel('iterations','fontsize',14)
ylabel('cost value','fontsize',14)
% print('-depsc','lambda_1')
xlim([1 max_iter])

set(gcf, 'PaperPosition', [0 0 7 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [7 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'data/simulation_results/large_2', 'pdf') %Save figure

