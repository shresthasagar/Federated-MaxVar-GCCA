%---------------------------------------------------
% Communication cost (bits per variable) vs cost value
% Iterations vs cost value
%---------------------------------------------------


% make sure that tensor_toolbox is in the matlab path

addpath src

clear;
clc;
close;

% Use the simulation filename in the format "scale_counter"
simulation_filename = 'small_1.mat'

TotalTrial = 1;
mu = .1;
noise_var = 0.1;
MaxIt = 600;
InnerIt = 10;
Nbits = 3;
r = .0001;

initial_comm_cost = 32;
per_iter_distr = Nbits;
per_iter_full_res = 32;

batch_size = [];
inner_iters = [1, 10];
max_iters = [10000, 10000];

for inner_it_idx=1:length(inner_iters)
    InnerIt = inner_iters(inner_it_idx);
    MaxIt = max_iters(inner_it_idx);
    for trial = 1:TotalTrial
        disp(['at trial ',num2str(trial)])
        I = 3;
        L = 50000;
        M = 5000;
        N = 5000;
        K = 10;
        m = 100;
        Z = sprandn(L, N, 1e-4);
        for i=1:I
            A{i}=sprandn(N,M, 0.0001);            
            X{i}=Z*A{i};
        end

        % MVLSA 
        % computing the global solution
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
        G = Um(:,order_ind(1:K));
        
        cost_global = 0;
        for i = 1:I 
            Q{i} = (1/sqrt(L))*(((1/L)*X{i}'*X{i}+r*eye(M))\(X{i}'*G));
            cost_global= (1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+cost_global+(r/2)*sum(sum(Q{i}.^2));
        end
        cost_optimal(trial)=cost_global;
        time_global = toc

        tic;
        filename = '../data/rand_1.mat';
        [ G_ini,Q_ini,Ux,Us,UB,cost_mlsa(trial),Li ] = MLSA( X,K,m,r);
        save(filename,'X','G_ini','Q_ini','cost_MLSA','Li');
        time_mlsa =toc;

        init_vars  = load(filename);
        X = init_vars.X;
        G_ini = init_vars.G_ini;
        Q_ini = init_vars.Q_ini;
        cost_MLSA = init_vars.cost_MLSA;
        Li = init_vars.Li;

        disp(['Full resolution with warm start ...'])
        [Q1,G_1,obj_warm_full_res{inner_it_idx}(trial,:),~,St1, t_warm_full_res] = LargeGCCA_distributed_stochastic( X,  K, ...
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
        comm_cost_iter_full_res{inner_it_idx}(1) = per_iter_full_res;
        for i=2:length(obj_warm_full_res{1}(1,:))
            comm_cost_iter_full_res{inner_it_idx}(i) = comm_cost_iter_full_res{inner_it_idx}(i-1) + per_iter_full_res;
        end


        % Ditributed MAX-VAR GCCA with warm start
        disp(['Distributed with warm start ...'])
        [Q1, G_1, obj_warm_distr{inner_it_idx}(trial,:), ~, St1, t_warm_distr] = LargeGCCA_distributed_stochastic(X,  K, ...
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
                                                                            'batch_size', 5000,  ...
                                                                            'rand_compress', true,  ...
                                                                            'compress_g', true, ...
                                                                            'print_log', true); 
        
        comm_cost_iter_distr{inner_it_idx}(1) = initial_comm_cost + per_iter_distr;
        for i=2:length(obj_warm_distr{inner_it_idx}(1,:))
            comm_cost_iter_distr{inner_it_idx}(i) = comm_cost_iter_distr{inner_it_idx}(i-1) + per_iter_distr;
        end 
        
    end  
end

%% Visualization of Communication cost (bytes) vs Iteration
figure(1)

h1 = loglog(comm_cost_iter_full_res{1}, squeeze(obj_warm_full_res{1}(1,:)), '-b', 'Linewidth', 2, 'DisplayName', 'AltMaxVar (T=1)'); hold on
h2 = loglog(comm_cost_iter_distr{1}, squeeze(obj_warm_distr{1}(1,:)), '--b', 'Linewidth', 2, 'DisplayName', 'Distributed AltMaxVar (T=1)'); hold on
h3 = loglog(comm_cost_iter_full_res{2}, squeeze(obj_warm_full_res{2}(1,:)), '-k', 'Linewidth', 2, 'DisplayName', 'AltMaxVar (T=10)'); hold on
h4 = loglog(comm_cost_iter_distr{2}, squeeze(obj_warm_distr{2}(1,:)), '--k', 'Linewidth', 2, 'DisplayName', 'Distributed AltMaxVar (T=10)'); hold off
% h5 = loglog(comm_cost_iter_full_res, squeeze(obj_warm_full_res(1,3,:)), '-r', 'Linewidth', 2); hold on
% h6 = loglog(comm_cost_iter_distr, squeeze(obj_warm_distr(1,3,:)), '--r', 'Linewidth', 2); hold off
% legend([h1,h2,h3,h4,h5,h6], {'AltMaxVar (T=1)', 'Distributed AltMaxVar (T=1)','AltMaxVar (T=10)', 'Distributed AltMaxVar (T=10)','AltMaxVar (T=100)', 'Distributed AltMaxVar (T=100)'})

legend
% x_limit = max(comm_cost_iter_distr(end), comm_cost_iter_full_res(end))
xlabel('communication cost (bits per variable)','fontsize',14)
ylabel('objective value','fontsize',14)
% print('-depsc','lambda_1')
% xlim([1 x_limit])

set(gcf, 'PaperPosition', [0 0 7 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [7 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, '../data/simulation_results/large_1_4', 'pdf') %Save figure

%% 
figure(2)
h1 = loglog(1:length(obj_warm_full_res{1}(1,:)), squeeze(obj_warm_full_res{1}(1,:)), '-b', 'Linewidth', 2, 'DisplayName', 'AltMaxVar (T=1)'); hold on
h2 = loglog(1:length(obj_warm_distr{1}(1,:)), squeeze(obj_warm_distr{1}(1,:)), '--b', 'Linewidth', 2, 'DisplayName', 'Distributed AltMaxVar (T=1)'); hold on
h3 = loglog(1:length(obj_warm_full_res{2}(1,:)), squeeze(obj_warm_full_res{2}(1,:)), '-k', 'Linewidth', 2, 'DisplayName', 'AltMaxVar (T=10)'); hold on
h4 = loglog(1:length(obj_warm_distr{2}(1,:)), squeeze(obj_warm_distr{2}(1,:)), '--k', 'Linewidth', 2, 'DisplayName', 'Distributed AltMaxVar (T=10)'); hold off
% h5 = loglog(comm_cost_iter_full_res, squeeze(obj_warm_full_res(1,3,:)), '-r', 'Linewidth', 2); hold on
% h6 = loglog(comm_cost_iter_distr, squeeze(obj_warm_distr(1,3,:)), '--r', 'Linewidth', 2); hold off
% legend([h1,h2,h3,h4,h5,h6], {'AltMaxVar (T=1)', 'Distributed AltMaxVar (T=1)','AltMaxVar (T=10)', 'Distributed AltMaxVar (T=10)','AltMaxVar (T=100)', 'Distributed AltMaxVar (T=100)'})

legend('location', 'southwest')
% x_limit = max(comm_cost_iter_distr(end), comm_cost_iter_full_res(end))
xlabel('iterations','fontsize',14)
ylabel('objective value','fontsize',14)
% print('-depsc','lambda_1')
% xlim([1 x_limit])

set(gcf, 'PaperPosition', [0 0 7 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [7 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, '../data/simulation_results/large_2_4', 'pdf') %Save figure

%% 
full_res = load('../data/simulation_conditions/large_full_res.mat');
distr = load('../data/simulation_conditions/large_distr.mat')

figure(3)
tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact');

t_steps = 3000;

nexttile
h1 = loglog(full_res.comm_cost_iter_full_res{1}(1:t_steps), squeeze(full_res.obj_warm_full_res{1}(1,1:t_steps)), '-b', 'Linewidth', 4, 'DisplayName', 'AltMaxVar (T=1)'); hold on
h2 = loglog(distr.comm_cost_iter_distr{1}(1:t_steps), squeeze(distr.obj_warm_distr{1}(1,1:t_steps)), '--b', 'Linewidth', 4, 'DisplayName', 'Distributed AltMaxVar (T=1)'); hold on
h3 = loglog(full_res.comm_cost_iter_full_res{2}(1:t_steps), squeeze(full_res.obj_warm_full_res{2}(1,1:t_steps)), '-k', 'Linewidth', 4, 'DisplayName', 'AltMaxVar (T=10)'); hold on
h4 = loglog(distr.comm_cost_iter_distr{2}(1:t_steps), squeeze(distr.obj_warm_distr{2}(1,1:t_steps)), '--k', 'Linewidth', 4, 'DisplayName', 'Distributed AltMaxVar (T=10)'); hold off

% legend
% x_limit = max(comm_cost_iter_distr(end), comm_cost_iter_full_res(end))
xlabel('bits per variable','fontsize',14)
ylabel('objective value','fontsize',14)
% print('-depsc','lambda_1')
% xlim([1 x_limit])
ax = gca;
ax.FontSize = 14;

nexttile
h1 = loglog(1:length(full_res.obj_warm_full_res{1}(1,1:t_steps)), squeeze(full_res.obj_warm_full_res{1}(1,1:t_steps)), '-b', 'Linewidth', 4, 'DisplayName', 'AltMaxVar (T=1)'); hold on
h2 = loglog(1:length(distr.obj_warm_distr{1}(1,1:t_steps)), squeeze(distr.obj_warm_distr{1}(1,1:t_steps)), '--b', 'Linewidth', 4, 'DisplayName', 'Distributed AltMaxVar (T=1)'); hold on
h3 = loglog(1:length(full_res.obj_warm_full_res{2}(1,1:t_steps)), squeeze(full_res.obj_warm_full_res{2}(1,1:t_steps)), '-k', 'Linewidth', 4, 'DisplayName', 'AltMaxVar (T=10)'); hold on
h4 = loglog(1:length(distr.obj_warm_distr{2}(1,1:t_steps)), squeeze(distr.obj_warm_distr{2}(1,1:t_steps)), '--k', 'Linewidth', 4, 'DisplayName', 'Distributed AltMaxVar (T=10)'); hold off

legend('location', 'southwest')
% x_limit = max(comm_cost_iter_distr(end), comm_cost_iter_full_res(end))
xlabel('iterations','fontsize',14)
% ylabel('objective value','fontsize',14)
% print('-depsc','lambda_1')
% xlim([1 x_limit])
ax = gca;
ax.FontSize = 14;

set(gcf, 'PaperPosition', [0 0 10 4]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [10 4]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'data/simulation_results/large_combined', 'pdf') %Save figure