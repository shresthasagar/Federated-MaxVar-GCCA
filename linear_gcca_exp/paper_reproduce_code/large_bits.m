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
simulation_filename = 'large_bits.mat'

% Commonly available data inside the simulation file
%   1. Objective value at each iteration
%   2. Elapsed time at each iteration
%   3. Data generating conditions (L, M, N, K, m, mu)

TotalTrial = 1;
mu = .1;
noise_var = 0.1;
MaxIt = 100;
InnerIt = 10;
Nbits = 3;
r = 0;
batch_size = 5000

folder_name = '../data/simulation_conditions_revised/large_bits/'

bits_list = [2, 3, 4, 5]

bits_list = [3]

for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])

    % dense: sanity check
    % I = 3;
    % L = 500;
    % Lbad = 0;
    % M = 25;
    % N = 20;
    % K = 5;
    % m = 10;
    % Z = randn(L,N);
    % for i=1:I
    %     A{i} = randn(N,M);
    %     X{i} = Z*A{i} + noise_var*randn(L,M); 
    %     condnum(i) = cond((1/L)*X{1}'*X{1});
    % end
    
    % cond_mean(trial)=mean( condnum(i));
    

    I = 3;
    L = 50000;
    M = 2000;
    N = 200;
    K = 5;
    m = 50;

    % denser Z seems to work better for sgd based compression in large scale. why?
    Z = sprandn(L, N, 0.01);%*diag(2+1*randn(N,1));
    % Z = randn(L, N);%*diag(2+1*randn(N,1));

    for i=1:I
        A{i}=sprandn(N, M, 5e-4);            
        X{i}=Z*A{i}+ noise_var*sprandn(L, M, 1e-3);
    end

    % cond_mean(trial)=mean( condnum(i));
    tic;
    [ G_ini,Q_ini,Ux,Us,UB,cost_mlsa(trial),Li ] = MLSA( X,K,m,r);
    % dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
    time_mlsa =toc;

    for idx=1:length(bits_list)
        Nbits = bits_list(idx)

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
                                                                                'print_log', true); 
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
                                                                            'compress_avg', false, ...
                                                                            'print_log', true); 

    
    
        
    end  
end


%% Visualization of Communication cost (bytes) vs Iteration
figure(1)
% save the output for plots in the future
filename = '../data/simulation_conditions_revised/large_bits.mat';
save(filename, 'obj_warm_full_res', 'obj_warm_distr', 't_warm_full_res', 't_warm_distr');

TotalTrial = 5;
mu = .1;
noise_var = 0.1;
MaxIt = 100;
InnerIt = 1;
Nbits = 3;
r = 0;
batch_size = 5000

folder_name = '../data/simulation_conditions_revised/large_bits/'

bits_list = [2, 3, 4, 5]
bits_list = [3]

large = load(filename);
obj_warm_full_res = large.obj_warm_full_res;
obj_warm_distr = large.obj_warm_distr;
t_warm_full_res = large.t_warm_full_res;
t_warm_distr = large.t_warm_distr;

initial_comm_cost = 32;

per_iter_distr = Nbits;
per_iter_full_res = 32;

for idx=1:length(bits_list)
    per_iter_distr = bits_list(idx);
    comm_cost_iter_distr(idx,1) = per_iter_full_res + per_iter_distr;
    for i=2:length(obj_warm_distr(1,1,:))
        comm_cost_iter_distr(idx, i) = comm_cost_iter_distr(idx, i-1) + per_iter_distr;
    end
end

comm_cost_iter_full_res(1) = 2*per_iter_full_res;
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

legend_text = {legend_text_full_res, legend_text_distr};
legend_text = cat(2, legend_text{:})


leg = legend([h_full_res h_distr(1:end)], legend_text, 'fontsize', 16, 'Orientation', 'Horizontal', 'Numcolumns', 3);
leg.Layout.Tile = 'north'

x_limit = max(comm_cost_iter_distr(end), comm_cost_iter_full_res(end))
xlabel('communication cost (BPV)','fontsize',16)
ylabel('objective value','fontsize',16)
xlim([min(comm_cost_iter_distr, [], 'all') max(comm_cost_iter_full_res, [], 'all')])
ax = gca;
ax.FontSize=16


nexttile 

h_full_res = loglog([1:MaxIt+1], squeeze(mean(obj_warm_full_res(:,1,:),1)), colors_full_res{length(bits_list)}, 'Linewidth', 2); hold on
legend_text_full_res = ['AltMaxVar, Full gradient']; 

for idx=1:length(bits_list)
    h_distr(idx) = loglog([1:MaxIt+1], squeeze(mean(obj_warm_distr(:,idx,:), 1)), colors_distr{idx}, 'Linewidth', 2); hold on
    legend_text_distr{idx} = ['CuteMaxVar, batch size:', num2str(bits_list(idx))];
end

xlabel('iteration','fontsize',16)

ax = gca;
ax.FontSize = 16;

set(gcf, 'PaperPosition', [0 0 12 6]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [12 6]); %Set the paper to have width 5 and height 5.
saveas(gcf, '../data/simulation_results_revised/large_bits_averaged_old2', 'pdf') %Save figure