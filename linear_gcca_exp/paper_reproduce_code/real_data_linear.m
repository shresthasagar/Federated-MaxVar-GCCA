addpath /scratch/sagar/Projects/matlab/lib/tensor_toolbox
addpath ../src

clear;
clc;
close;

TotalTrial = 1;
noise_var = 0.1;
MaxIt = 2600;
InnerIt = 10;
Nbits = 3;


data = load('../../real_data/Europarl_all.mat');
[~, I] = size(data.data);
I = 3;

for i=1:I 
    X{i} = data.data{i};
end

[L, M] = size(X{1});
N = 50000;
K = 10;
m = 100;
r = 0;

%% How to initialize is another problem...
    
tic;
filename = ['../data/real_data_mvlsa_3view.mat'];

% [ G_ini,Q_ini,Ux,Us,UB,cost_MLSA,Li ] = MLSA( X,K,m,r);

% save(filename,'X','G_ini','Q_ini','cost_MLSA','Li');

toc;

% load from file
init_vars  = load(filename);

X = init_vars.X;
G_ini = init_vars.G_ini;
Q_ini = init_vars.Q_ini;
cost_MLSA = init_vars.cost_MLSA;
Li = init_vars.Li;


for trial = 1:TotalTrial


    % [Q1,G1,obj_full_res_gd(trial, :),~,St1, t_full_res_gd(trial, :)] = LargeGCCA_distributed_stochastic( X,  K, ...
    %                                                                     'G_ini',G_ini, ...
    %                                                                     'Q_ini',Q_ini, ...
    %                                                                     'r',r, ...
    %                                                                     'algo_type','plain', ...
    %                                                                     'Li',Li, ...
    %                                                                     'MaxIt',4000, ...
    %                                                                     'Inner_it',InnerIt, ...
    %                                                                     'Reg_type', 'none',  ...
    %                                                                     'distributed', false,  ...
    %                                                                     'nbits', Nbits,  ...
    %                                                                     'sgd', true,  ...
    %                                                                     'batch_size', 2000,  ...
    %                                                                     'rand_compress', false,  ...
    %                                                                     'compress_g', false, ...
    %                                                                     'print_log', true);
    % save('../data/simulation_conditions/real_optimal_altmaxvar.mat', 'Q1', 'G1', 't_full_res_gd', 'obj_full_res_gd')

    
    [Q1,G1,obj_full_res_gd(trial, :),~,St1, t_full_res_gd(trial, :)] = LargeGCCA_distributed_stochastic( X,  K, ...
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
                                                                        'batch_size', 2000,  ...
                                                                        'rand_compress', false,  ...
                                                                        'compress_g', false, ...
                                                                        'print_log', true);
    save('../data/simulation_conditions/real_data_full_res_full_gd_optimal.mat', 'Q1', 'G1', 't_full_res_gd', 'obj_full_res_gd')
    
    % [Q1,G1,obj_full_res_sgd(trial, :),~,St1, t_full_res_sgd(trial, :)] = LargeGCCA_distributed_stochastic( X,  K, ...
    %                                                                     'G_ini',G_ini, ...
    %                                                                     'Q_ini',Q_ini, ...
    %                                                                     'r',r, ...
    %                                                                     'algo_type','plain', ...
    %                                                                     'Li',Li, ...
    %                                                                     'MaxIt',MaxIt, ...
    %                                                                     'Inner_it',InnerIt, ...
    %                                                                     'Reg_type', 'none',  ...
    %                                                                     'distributed', false,  ...
    %                                                                     'nbits', Nbits,  ...
    %                                                                     'sgd', true,  ...
    %                                                                     'batch_size', 2000,  ...
    %                                                                     'rand_compress', false,  ...
    %                                                                     'compress_g', false, ...
    %                                                                     'print_log', true);
    % save('../data/simulation_conditions/real_data_full_res_sgd_optimal.mat', 'Q1', 'G1', 't_full_res_sgd', 'obj_full_res_sgd')

    
    
    % [Q2,G2,obj_gd(trial, :),~,St1, t_gd(trial, :)] = LargeGCCA_distributed_stochastic( X,  K, ...
    %                                                                     'G_ini',G_ini, ...
    %                                                                     'Q_ini',Q_ini, ...
    %                                                                     'r',r, ...
    %                                                                     'algo_type','plain', ...
    %                                                                     'Li',Li, ...
    %                                                                     'MaxIt',MaxIt, ...
    %                                                                     'Inner_it',InnerIt, ...
    %                                                                     'Reg_type', 'none',  ...
    %                                                                     'distributed', true,  ...
    %                                                                     'nbits', Nbits,  ...
    %                                                                     'sgd', false,  ...
    %                                                                     'batch_size', 2000,  ...
    %                                                                     'rand_compress', true,  ...
    %                                                                     'compress_g', true, ...
    %                                                                     'print_log', true);
    % save('../data/simulation_conditions/real_data_distr_full_gd_optimal.mat', 'Q2', 'G2', 't_gd', 'obj_gd')

    % [Q3,G3,obj_sgd(trial, :),~,St1, t_sgd(trial, :)] = LargeGCCA_distributed_stochastic( X,  K, ...
    %                                                                     'G_ini',G_ini, ...
    %                                                                     'Q_ini',Q_ini, ...
    %                                                                     'r',r, ...
    %                                                                     'algo_type','plain', ...
    %                                                                     'Li',Li, ...
    %                                                                     'MaxIt',MaxIt, ...
    %                                                                     'Inner_it',InnerIt, ...
    %                                                                     'Reg_type', 'none',  ...
    %                                                                     'distributed', true,  ...
    %                                                                     'nbits', Nbits,  ...
    %                                                                     'sgd', true,  ...
    %                                                                     'batch_size', 2000,  ...
    %                                                                     'rand_compress', true,  ...
    %                                                                     'compress_g', true, ...
    %                                                                     'print_log', true);
    % save('../data/simulation_conditions/real_data_distr_sgd_optimal.mat', 'Q3', 'G3', 't_sgd', 'obj_sgd')
end
%% 
%% 
Nbits = 3
full_res_gd = load('../data/simulation_conditions/real_data_full_res_full_gd_trials.mat')
full_res_sgd = load('../data/simulation_conditions/real_data_full_res_sgd_trials.mat')
gd = load('../data/simulation_conditions/real_data_distr_full_gd_trials.mat')
sgd = load('../data/simulation_conditions/real_data_distr_sgd_trials.mat')

%%
initial_comm_cost = 32;
per_iter_distr = Nbits;
per_iter_full_res = 32;


comm_cost_full_res(1) = initial_comm_cost + per_iter_full_res;
for i=2:length(full_res_gd.obj_full_res_gd)
    comm_cost_full_res(i) = comm_cost_full_res(i-1) + per_iter_full_res;
end                                                                                


comm_cost_gd(1) = initial_comm_cost + per_iter_distr;
for i=2:length(gd.obj_gd)
    comm_cost_gd(i) = comm_cost_gd(i-1) + per_iter_distr;
end                                                                                


comm_cost_sgd(1) = initial_comm_cost + per_iter_distr;
for i=2:length(sgd.obj_sgd)
    comm_cost_sgd(i) = comm_cost_sgd(i-1) + per_iter_distr;
end

t_steps = length(sgd.obj_sgd);
figure(1)
tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact');

nexttile
h1 = plot(comm_cost_full_res(1:t_steps), mean(full_res_gd.obj_full_res_gd, 1), '-b', 'Linewidth', 2, 'DisplayName', 'AltMaxVar, full gradient'); hold on
h4 = plot(comm_cost_full_res(1:t_steps), mean(full_res_sgd.obj_full_res_sgd, 1), '--g', 'Linewidth', 2, 'DisplayName', 'AltMaxVar, full gradient'); hold on
h2 = plot(comm_cost_gd(1:t_steps),  mean(gd.obj_gd, 1), '-r', 'Linewidth', 2, 'DisplayName', 'CuteMaxVar, full gradient'); hold on
h3 = plot(comm_cost_sgd(1:t_steps), mean(sgd.obj_sgd, 1), '--k', 'Linewidth', 2, 'DisplayName', 'CuteMaxVar, batch size: 2000'); hold on

xlabel('communication cost (BPV)', 'fontsize',16)
ylabel('objective value','fontsize',16)
ax = gca;
ax.FontSize = 16;

nexttile;
h1 = plot(mean(full_res_gd.t_full_res_gd, 1), mean(full_res_gd.obj_full_res_gd, 1), '-b', 'Linewidth', 2, 'DisplayName', 'AltMaxVar, full gradient'); hold on
h4 = plot(mean(full_res_sgd.t_full_res_sgd, 1), mean(full_res_sgd.obj_full_res_sgd, 1), '--g', 'Linewidth', 2, 'DisplayName', 'AltMaxVar batch size: 2000'); hold on
h2 = plot(mean(gd.t_gd, 1), mean(gd.obj_gd, 1), '-r', 'Linewidth', 2, 'DisplayName', 'CuteMaxVar full gradient'); hold on
h3 = plot(mean(sgd.t_sgd, 1), mean(sgd.obj_sgd, 1), '--k', 'Linewidth', 2, 'DisplayName', 'CuteMaxVar batch size: 2000'); hold on

legend
xlabel('time (seconds)', 'fontsize',16)

ax = gca;
ax.FontSize = 16;

set(gcf, 'PaperPosition', [0 0 15 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, '../data/simulation_results/real_averaged', 'pdf') %Save figure
