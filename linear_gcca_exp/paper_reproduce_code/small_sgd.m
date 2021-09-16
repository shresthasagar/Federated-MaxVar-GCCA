% A simple script to show that local is global
% min_{G Q_i} sum_i||X_i*Q_i-G||
% st G^tG = I;
% X_i is tall and full rank

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

TotalTrial = 50;
mu = .1;
noise_var = 0.1;
MaxIt = 1000;
InnerIt = 10;
Nbits = 3;
r = 0;



batch_sizes = [10, 50, 100, 200, 350, 500]


for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])
    
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

    % cond_mean(trial)=mean( condnum(i));
    tic;
    [ G_ini,Q_ini,Ux,Us,UB,cost_mlsa(trial),Li ] = MLSA( X,K,m,r);
    % dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
    time_mlsa =toc;


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
    % tic;
    % [ G_ini,Q_ini,Ux,Us,UB,cost_mlsa(trial),Li ] = MLSA( X,K,m,r);
    % % dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
    % time_mlsa =toc;
    
    %% Full resolution MAX-VAR GCCA with warm start
    for batch_size_idx=1:length(batch_sizes)
        batch_size = batch_sizes(batch_size_idx)

        if batch_size == L
            disp(['Full resolution with warm start ...'])
            [Q1, G_1, obj_warm_full_res(trial, batch_size_idx, :), ~, St1, t_warm_full_res(trial, batch_size_idx, :)] = LargeGCCA_distributed_stochastic( X,  K, ...
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
                                                                                'sgd', true,  ...
                                                                                'batch_size', batch_size,  ...
                                                                                'rand_compress', false,  ...
                                                                                'compress_g', false, ...
                                                                                'print_log', false); 
        end
                                                          
        % Ditributed MAX-VAR GCCA with warm start
        disp(['Distributed with warm start ...'])
        [Q1,G_1,obj_warm_distr(trial,batch_size_idx,:),~,St1, t_warm_distr(trial, batch_size_idx, :)] = LargeGCCA_distributed_stochastic( X,  K, ...
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
                                                                            'sgd', true,  ...
                                                                            'batch_size', batch_size,  ...
                                                                            'rand_compress', true,  ...
                                                                            'compress_g', true, ...
                                                                            'print_log', false); 

    
    
        
    end  
end

%% save the output for plots in the future
filename = '../data/simulation_conditions/small_sgd.mat';
% save(filename, 'obj_warm_full_res', 'obj_warm_distr', 't_warm_full_res', 't_warm_distr', 'cost_optimal');


small = load(filename);
obj_warm_full_res = small.obj_warm_full_res;
obj_warm_distr = small.obj_warm_distr;
t_warm_full_res = small.t_warm_full_res;
t_warm_distr = small.t_warm_distr;
cost_optimal = small.cost_optimal;


%% Visualization of Communication cost (bytes) vs Iteration
figure(1)


TotalTrial = 50;
mu = .1;
noise_var = 0.1;
MaxIt = 1000;
InnerIt = 10;
Nbits = 3;
r = 0;
L = 500


batch_sizes = [10, 50, 100, 200, 350, 500]

initial_comm_cost = 32;

per_iter_distr = Nbits;
per_iter_full_res = 32;

comm_cost_iter_distr(1) = per_iter_full_res + per_iter_distr;
for i=2:length(obj_warm_distr(1,1,:))
    comm_cost_iter_distr(i) = comm_cost_iter_distr(i-1) + per_iter_distr;
end

comm_cost_iter_full_res(1) = 2*per_iter_full_res;
for i=2:length(obj_warm_full_res(1,1,:))
    comm_cost_iter_full_res(i) = comm_cost_iter_full_res(i-1) + per_iter_full_res;
end

colors_full_res = {'-b', '-k', '-r', '-m', '-c', 'g'};
colors_distr = {'--b', '--k', '--r', '--m', '--c', '--g'};


tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact');

nexttile

h_full_res = loglog(comm_cost_iter_full_res, squeeze(mean(obj_warm_full_res(:,length(batch_sizes),:), 1)), colors_full_res{length(batch_sizes)}, 'Linewidth', 2); hold on
legend_text_full_res = ['AltMaxVar, Full gradient'];

for batch_size_idx=1:length(batch_sizes)
    h_distr(batch_size_idx) = loglog(comm_cost_iter_distr, squeeze(mean(obj_warm_distr(:,batch_size_idx,:), 1)), colors_distr{batch_size_idx}, 'Linewidth', 2); hold on
    legend_text_distr{batch_size_idx} = ['CuteMaxVar, batch size:', num2str(batch_sizes(batch_size_idx))];
end
% Optimal

max_comm = max(comm_cost_iter_full_res)
h_opt = loglog(linspace(min(comm_cost_iter_distr),max_comm, length(comm_cost_iter_full_res)), mean(cost_optimal)*ones(1, length(comm_cost_iter_full_res)), '-b', 'linewidth', 2); hold on;


if batch_sizes(end) == L
    legend_text_distr{length(batch_sizes)} = ['CuteMaxVar, Full gradient']; 
end
legend_text_distr{end+1} = ['optimal'];
legend_text = {legend_text_full_res, legend_text_distr, };
legend_text = cat(2, legend_text{:})

leg = legend([h_full_res h_distr(1:end) h_opt], legend_text, 'fontsize', 16, 'Orientation', 'Horizontal', 'Numcolumns', 3 );
leg.Layout.Tile = 'north'

x_limit = max(comm_cost_iter_distr(end), comm_cost_iter_full_res(end))
xlabel('communication cost (BPV)','fontsize',16)
ylabel('objective value','fontsize',16)
% print('-depsc','lambda_1')
xlim([min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all')]);
ax = gca;
ax.FontSize=16

% set(gcf, 'PaperPosition', [0 0 7 5]); %Position plot at left hand corner with width 5 and height 5.
% set(gcf, 'PaperSize', [7 5]); %Set the paper to have width 5 and height 5.
% saveas(gcf, '../data/simulation_results/small_sgd_comm2', 'pdf') %Save figure



nexttile 

h_full_res = loglog(squeeze(mean(t_warm_full_res(:,length(batch_sizes), :), 1)), squeeze(mean(obj_warm_full_res(:,length(batch_sizes),:), 1)), colors_full_res{batch_size_idx}, 'Linewidth', 2); hold on
legend_text_full_res = ['AltMaxVar, Full gradient']; 

for batch_size_idx=1:length(batch_sizes)
    h_distr(batch_size_idx) = loglog(squeeze(mean(t_warm_distr(:,batch_size_idx,:), 1)), squeeze(mean(obj_warm_distr(:,batch_size_idx,:), 1)), colors_distr{batch_size_idx}, 'Linewidth', 2); hold on
    legend_text_distr{batch_size_idx} = ['CuteMaxVar, batch size:', num2str(batch_sizes(batch_size_idx))];
end
% max_iter = max(t_warm_full_res, [], 'all');
h_opt = loglog(linspace(min(t_warm_distr(:,:,2:end), [], 'all'), max(t_warm_full_res, [], 'all'), MaxIt+1), mean(cost_optimal)*ones(1, MaxIt+1), '-b', 'linewidth', 2); hold on;

% if batch_sizes(end) == L
%     legend_text_distr{length(batch_sizes)} = ['CuteMaxVar, Full gradient']; 
% end
% legend_text = {legend_text_full_res, legend_text_distr};
% legend_text = cat(2, legend_text{:})

% legend([h_full_res h_distr(1:end)], legend_text, 'fontsize', 16 )

xlabel('time (seconds)','fontsize',16)
% ylabel('objective value','fontsize',16)
% print('-depsc','lambda_1')
xlim([min(t_warm_distr(:,:,2:end), [], 'all') 1])
ax = gca;
ax.FontSize = 16;

set(gcf, 'PaperPosition', [0 0 12 6]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [12 6]); %Set the paper to have width 5 and height 5.
saveas(gcf, '../data/simulation_results/small_sgd_averaged', 'pdf') %Save f