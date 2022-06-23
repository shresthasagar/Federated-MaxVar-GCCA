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
simulation_filename = 'small_bits.mat'

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
batch_size = 150



bits_list = [2, 3, 4, 5]
for idx=1:length(bits_list)
    Nbits = bits_list(idx)
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
        [ G_ini,Q_ini,Ux,Us,UB,cost_mlsa,Li ] = MLSA( X,K,m,r);
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
            cost_global = (1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+cost_global+(r/2)*sum(sum(Q{i}.^2));
        end
        cost_optimal(trial)=cost_global;
        time_global = toc
        
        %% MVLSA 
        % tic;
        % [ G_ini,Q_ini,Ux,Us,UB,cost_mlsa(trial),Li ] = MLSA( X,K,m,r);
        % % dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
        % time_mlsa =toc;
        
        %% Full resolution MAX-VAR GCCA with warm start

        if Nbits == bits_list(1)
            disp(['Full resolution with warm start ...'])
            [Q1, G_1, obj_warm_full_res(trial, idx, :), ~, St1, t_warm_full_res(trial, idx, :)] = LargeGCCA_distributed_stochastic( X,  K, ...
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
        [Q1,G_1,obj_warm_distr(trial,idx,:),~,St1, t_warm_distr(trial, idx, :)] = LargeGCCA_distributed_stochastic( X,  K, ...
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


%% Visualization of Communication cost (bytes) vs Iteration
figure(1)

TotalTrial = 50;
mu = .1;
noise_var = 0.1;
MaxIt = 1000;
InnerIt = 10;
Nbits = 3;
r = 0;
batch_size = 150



bits_list = [2, 3, 4, 5]
% save the output for plots in the future
filename = '../data/simulation_conditions/small_bits.mat';
save(filename, 'obj_warm_full_res', 'obj_warm_distr', 't_warm_full_res', 't_warm_distr', 'cost_optimal');


small = load(filename);
obj_warm_full_res = small.obj_warm_full_res;
obj_warm_distr = small.obj_warm_distr;
t_warm_full_res = small.t_warm_full_res;
t_warm_distr = small.t_warm_distr;
cost_optimal = small.cost_optimal;

initial_comm_cost = 32;

per_iter_distr = Nbits;
per_iter_full_res = 32;

for idx=1:length(bits_list)
    per_iter_distr = bits_list(idx);
    comm_cost_iter_distr(idx,1) = per_iter_full_res+per_iter_distr;
    for i=2:length(obj_warm_distr(1,1,:))
        comm_cost_iter_distr(idx, i) = comm_cost_iter_distr(idx, i-1) + per_iter_distr;
    end
end

comm_cost_iter_full_res(1) = per_iter_full_res*2;
for i=2:length(obj_warm_full_res(1,1,:))
    comm_cost_iter_full_res(i) = comm_cost_iter_full_res(i-1) + per_iter_full_res;
end

colors_full_res = {'-b', '-k', '-r', '-m', '-c', 'g'};
colors_distr = {'--b', '--k', '--r', '--m', '--c', '--g'};


tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact');

nexttile

h_full_res = loglog(comm_cost_iter_full_res, squeeze(mean(obj_warm_full_res(:,1,:), 1)), colors_full_res{length(bits_list)}, 'Linewidth', 2); hold on
legend_text_full_res = ['AltMaxVar'];

for idx=1:length(bits_list)
    h_distr(idx) = loglog(comm_cost_iter_distr(idx,:), squeeze(mean(obj_warm_distr(:,idx,:),1)), colors_distr{idx}, 'Linewidth', 2); hold on
    legend_text_distr{idx} = ['CuteMaxVar, ', num2str(bits_list(idx)), ' bits'];
end
h_opt = loglog(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_full_res)), mean(cost_optimal)*ones(1, length(comm_cost_iter_full_res)), '-b', 'linewidth', 2); hold on;
legend_text_distr{end+1} = ['optimal'];



legend_text = {legend_text_full_res, legend_text_distr};
legend_text = cat(2, legend_text{:})


leg = legend([h_full_res h_distr(1:end) h_opt], legend_text, 'fontsize', 16, 'Orientation', 'Horizontal', 'Numcolumns', 3);
leg.Layout.Tile = 'north'

x_limit = max(comm_cost_iter_distr(end), comm_cost_iter_full_res(end))
xlabel('communication cost (BPV)','fontsize',16)
ylabel('objective value','fontsize',16)
% print('-depsc','lambda_1')
xlim([min(comm_cost_iter_distr, [], 'all') max(comm_cost_iter_full_res, [], 'all')])
ax = gca;
ax.FontSize=16

% set(gcf, 'PaperPosition', [0 0 7 5]); %Position plot at left hand corner with width 5 and height 5.
% set(gcf, 'PaperSize', [7 5]); %Set the paper to have width 5 and height 5.
% saveas(gcf, '../data/simulation_results/small_sgd_comm2', 'pdf') %Save figure



nexttile 

h_full_res = loglog([1:MaxIt+1], squeeze(mean(obj_warm_full_res(:,1,:),1)), colors_full_res{length(bits_list)}, 'Linewidth', 2); hold on
legend_text_full_res = ['AltMaxVar, Full gradient']; 

for idx=1:length(bits_list)
    h_distr(idx) = loglog([1:MaxIt+1], squeeze(mean(obj_warm_distr(:,idx,:), 1)), colors_distr{idx}, 'Linewidth', 2); hold on
    legend_text_distr{idx} = ['CuteMaxVar, batch size:', num2str(bits_list(idx))];
end
h_opt = loglog(1:MaxIt+1, mean(cost_optimal)*ones(1, MaxIt+1), '-b', 'linewidth', 2); hold on;

% if bits_list(end) == L
%     legend_text_distr{length(bits_list)} = ['CuteMaxVar, Full gradient']; 
% end
% legend_text = {legend_text_full_res, legend_text_distr};
% legend_text = cat(2, legend_text{:})

% legend([h_full_res h_distr(1:end)], legend_text, 'fontsize', 16 )

xlabel('iteration','fontsize',16)

ax = gca;
ax.FontSize = 16;

set(gcf, 'PaperPosition', [0 0 12 6]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [12 6]); %Set the paper to have width 5 and height 5.
saveas(gcf, '../data/simulation_results/small_bits_averaged', 'pdf') %Save figure


%% plot the optimal reference lines
filename = '../data/simulation_conditions/small_bits.mat';
% save(filename, 'obj_warm_full_res', 'obj_warm_distr', 't_warm_full_res', 't_warm_distr', 'cost_optimal');
bits_list = [2, 3, 4, 5]

small = load(filename);
obj_warm_full_res = mean(small.obj_warm_full_res, 1);
obj_warm_distr = mean(small.obj_warm_distr, 1);
t_warm_full_res = mean(small.t_warm_full_res, 1);
t_warm_distr = mean(small.t_warm_distr, 1);
cost_optimal = mean(small.cost_optimal);

threshold = 4*cost_optimal;

for i=1:length(obj_warm_full_res(1,1,:))
    if obj_warm_full_res(1,1,i) < threshold
        full_res_count = i
        break
    end
end
    
distr_cr = zeros(1, length(obj_warm_distr(1,:,1)));
for j=1:length(bits_list)
    for i=1:length(obj_warm_full_res(1,1,:))
        if obj_warm_distr(1,j,i) < threshold
            distr_cr(1,j) = 1 - (i*bits_list(j)/(full_res_count*32))
            break
        end
    end
end




%% Visualization of Communication cost (bytes) vs Iteration
figure(1)


initial_comm_cost = 32;

per_iter_full_res = 32;

for idx=1:length(bits_list)
    per_iter_distr = bits_list(idx);
    comm_cost_iter_distr(idx,1) = per_iter_full_res + per_iter_distr;
    for i=2:length(obj_warm_distr(1,1,:))
        comm_cost_iter_distr(idx, i) = comm_cost_iter_distr(idx, i-1) + per_iter_distr;
    end
end

optimal_line = cost_optimal*ones(1, length(comm_cost_iter_distr(2,:)));

h_distr = loglog(comm_cost_iter_distr(2,:), squeeze(obj_warm_distr(1, 2,:)), '--k', 'Linewidth', 2); hold on
h_opt = loglog(linspace(min(comm_cost_iter_distr(2,:), [], 'all'), max(comm_cost_iter_distr(2,:), [], 'all'), length(comm_cost_iter_distr(2,:))), optimal_line, '-b', 'linewidth', 2); hold on;
h_opt1 = loglog(linspace(min(comm_cost_iter_distr(2,:), [], 'all'), max(comm_cost_iter_distr(2,:), [], 'all'), length(comm_cost_iter_distr(2,:))), optimal_line*1.5, '-r', 'linewidth', 1); hold on;
h_opt2 = loglog(linspace(min(comm_cost_iter_distr(2,:), [], 'all'), max(comm_cost_iter_distr(2,:), [], 'all'), length(comm_cost_iter_distr(2,:))), optimal_line*2, '-r', 'linewidth', 1); hold on;
h_opt3 = loglog(linspace(min(comm_cost_iter_distr(2,:), [], 'all'), max(comm_cost_iter_distr(2,:), [], 'all'), length(comm_cost_iter_distr(2,:))), optimal_line*4, '-r', 'linewidth', 1); hold on;

legend_text{2} = 'optimal'
legend_text{1} = 'CuteMaxVar, 3 bits'

t1 = text(comm_cost_iter_distr(2,100), cost_optimal, 'optimal', 'fontsize', 14, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');
t2 = text(comm_cost_iter_distr(2,100), 1.5*cost_optimal, '1.5 x optimal, CR = 0.9062', 'fontsize', 14, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');
t3 = text(comm_cost_iter_distr(2,100), 2*cost_optimal, '2 x optimal, CR = 0.9062', 'fontsize', 14, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');
t4 = text(comm_cost_iter_distr(2,100), 4*cost_optimal, '4 x optimal, CR = 0.9062', 'fontsize', 14, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');

leg = legend([h_distr h_opt], legend_text, 'fontsize', 14);

xlabel('communication cost (BPV)','fontsize',14)
ylabel('objective value','fontsize',14)
% print('-depsc','lambda_1')
% xlim([min(comm_cost_iter_distr, [], 'all') max(comm_cost_iter_full_res, [], 'all')])

xlim([min(comm_cost_iter_distr(2,:), [], 'all'), max(comm_cost_iter_distr(2,:), [], 'all')]);
ylim([0.0015 0.01]);
ax = gca;
ax.FontSize=14

set(gcf, 'PaperPosition', [0 0 8 6]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [8 6]); %Set the paper to have width 5 and height 5.
saveas(gcf, '../data/simulation_results/small_bits_optimal_compare', 'pdf') %Save figure
