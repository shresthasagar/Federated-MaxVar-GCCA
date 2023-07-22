figure(1)
distr_eval = load('../data/simulation_outputs/real_data_sgd_distr_output.mat');
full_res_eval = load('../data/simulation_outputs/real_data_sgd_full_res_output.mat');

distr_obj = load('../data/simulation_outputs/real_data_sgd_distr_objective.mat');
full_res_obj = load('../data/simulation_outputs/real_data_sgd_full_res_objective.mat');

optimal = load('../data/simulation_conditions/real_optimal_altmaxvar.mat');
Nbits= 3;

obj_warm_full_res = full_res_obj.obj;
obj_warm_distr = distr_obj.obj;

aroc_full_res = full_res_eval.aroc();
nn_freq_full_res = full_res_eval.nn_freq;

aroc_distr = distr_eval.aroc(1:length(aroc_full_res));
nn_freq_distr = distr_eval.nn_freq(1:length(nn_freq_full_res));

cost_optimal = optimal.obj_full_res_gd(1,end);


tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact');

nexttile
% threshold = 1.5*cost_optimal;
thresh = [1.5, 2];

obj_full_res_count = zeros(1, length(thresh));
for j=1:length(thresh)

    for i=1:length(obj_warm_full_res)
        if obj_warm_full_res(i) < thresh(j)*cost_optimal
            full_res_count(j) = i;
            break
        end
    end
end
    

distr_cr = zeros(1, length(thresh));
for j=1:length(thresh)
    for i=1:length(obj_warm_full_res)
        if obj_warm_distr(i) < thresh(j)*cost_optimal
            distr_cr(1,j) = 1 - (i*Nbits/(full_res_count(1,j)*32))
            break
        end
    end
end



initial_comm_cost = 32;

per_iter_full_res = 32;

per_iter_distr = Nbits;
comm_cost_iter_distr(1) = per_iter_full_res + per_iter_distr;
comm_cost_iter_full_res(1) = 2*per_iter_full_res;
for i=2:length(obj_warm_distr)
    comm_cost_iter_full_res(i) = comm_cost_iter_full_res(i-1) + per_iter_full_res;
    comm_cost_iter_distr(i) = comm_cost_iter_distr(i-1) + per_iter_distr;
end

optimal_line = cost_optimal*ones(1, length(comm_cost_iter_distr));

h_full_res = plot(comm_cost_iter_full_res, obj_warm_full_res, '-r', 'Linewidth', 2); hold on
h_distr = plot(comm_cost_iter_distr, obj_warm_distr, '--r', 'Linewidth', 2); hold on
h_opt = plot(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_distr) ), optimal_line, '-b', 'linewidth', 2); hold on;
h_opt1 = loglog(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_distr)), optimal_line*1.5, '-c', 'linewidth', 2); hold on;
h_opt2 = loglog(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_distr)), optimal_line*2, '-c', 'linewidth', 2); hold on;


legend_text{2} = 'CuteMaxVar, 3 bits'
legend_text{1} = 'AltMaxVar'
% legend_text{3} = 'optimal'

t1 = text(comm_cost_iter_full_res(1500), cost_optimal, 'optimal', 'fontsize', 16, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');
t2 = text(comm_cost_iter_full_res(1500), 1.5*cost_optimal, '1.5 x optimal, CR = 0.9064', 'fontsize', 16, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');
t3 = text(comm_cost_iter_full_res(1500), 2*cost_optimal, '2 x optimal, CR = 0.9061', 'fontsize', 16, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');

% leg = legend([h_full_res h_distr h_opt], legend_text, 'fontsize', 16);
leg = legend([h_full_res h_distr], legend_text, 'fontsize', 16, 'Orientation', 'Horizontal', 'Numcolumns', 3);
leg.Layout.Tile = 'north'

xlabel('communication cost (BPV)','fontsize',16)
ylabel('objective value','fontsize',16)
% print('-depsc','lambda_1')
xlim([min(comm_cost_iter_distr, [], 'all') max(comm_cost_iter_full_res, [], 'all')])

% xlim([min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_distr, [], 'all')]);
ylim([0.15 0.8]);
ax = gca;
ax.FontSize=16


nexttile

thresh = [60 55]

nn_freq_full_res_count = zeros(1, length(thresh));
for j=1:length(thresh)
    for i=1:length(nn_freq_full_res)
        if nn_freq_full_res(i) < thresh(j)
            nn_freq_full_res_count(j) = i;
            break
        end
    end
end
    

nn_freq_distr_cr = zeros(1, length(thresh));
for j=1:length(thresh)
    for i=1:length(nn_freq_distr)
        if nn_freq_distr(i) < thresh(j)
            nn_freq_distr_cr(1,j) = 1 - (i*Nbits/(nn_freq_full_res_count(j)*32))
            break
        end
    end
end


initial_comm_cost = 32;
per_iter_full_res = 32*50;
per_iter_distr = Nbits*50;

comm_cost_iter_distr = 0;
comm_cost_iter_full_res = 0;

comm_cost_iter_distr(1) = per_iter_full_res + per_iter_distr;
comm_cost_iter_full_res(1) = 2*per_iter_full_res;
for i=2:length(nn_freq_full_res)
    comm_cost_iter_full_res(i) = comm_cost_iter_full_res(i-1) + per_iter_full_res;
    comm_cost_iter_distr(i) = comm_cost_iter_distr(i-1) + per_iter_distr;
end

thresh1 = thresh(1)*ones(1, length(comm_cost_iter_distr));
thresh2 = thresh(2)*ones(1, length(comm_cost_iter_distr));


h_full_res = plot(comm_cost_iter_full_res, nn_freq_full_res, '-r', 'Linewidth', 2); hold on
h_distr = plot(comm_cost_iter_distr, nn_freq_distr, '--r', 'Linewidth', 2); hold on
% h_opt = plot(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_distr) ), optimal_line, '-b', 'linewidth', 2); hold on;
h_opt1 = loglog(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_distr)), thresh1, '-c', 'linewidth', 2); hold on;
h_opt2 = loglog(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_distr)), thresh2, '-c', 'linewidth', 2); hold on;


legend_text{2} = 'CuteMaxVar, 3 bits'
legend_text{1} = 'AltMaxVar'
legend_text{3} = 'optimal'

% t1 = text(comm_cost_iter_full_res(1500), cost_optimal, 'optimal', 'fontsize', 16, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');
t2 = text(comm_cost_iter_full_res(20), thresh(1), 'CR = 0.9062', 'fontsize', 16, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');
t3 = text(comm_cost_iter_full_res(20), thresh(2), 'CR = 0.9062', 'fontsize', 16, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');


xlabel('communication cost (BPV)','fontsize',16)
ylabel('NN\_freq','fontsize',16)

xlim([min(comm_cost_iter_distr, [], 'all') max(comm_cost_iter_full_res, [], 'all')])
ylim([30 65]);
ax = gca;
ax.FontSize=16

set(gcf, 'PaperPosition', [0 0 12 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [12 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, '../data/simulation_results/real_test_train3', 'pdf') %Save figure






















% figure(1)
% distr = load('../data/simulation_conditions/real_data_distr_sgd_optimal.mat');
% full_res = load('../data/simulation_conditions/real_data_full_res_sgd_optimal.mat');
% optimal = load('../data/simulation_conditions/real_optimal_altmaxvar.mat');
% Nbits= 3;

% obj_warm_full_res = mean(full_res.obj_full_res_sgd, 1);
% obj_warm_distr = mean(distr.obj_sgd, 1);
% cost_optimal = optimal.obj_full_res_gd(1,end);

% % threshold = 1.5*cost_optimal;
% thresh = [1.5, 2];

% obj_full_res_count = zeros(1, length(thresh));
% for j=1:length(thresh)

%     for i=1:length(obj_warm_full_res)
%         if obj_warm_full_res(i) < thresh(j)*cost_optimal
%             full_res_count(j) = i;
%             break
%         end
%     end
% end
    

% distr_cr = zeros(1, length(thresh));
% for j=1:length(thresh)
%     for i=1:length(obj_warm_full_res)
%         if obj_warm_distr(i) < thresh(j)*cost_optimal
%             distr_cr(1,j) = 1 - (i*Nbits/(full_res_count(1,j)*32))
%             break
%         end
%     end
% end



% initial_comm_cost = 32;

% per_iter_full_res = 32;

% per_iter_distr = Nbits;
% comm_cost_iter_distr(1) = per_iter_full_res + per_iter_distr;
% comm_cost_iter_full_res(1) = 2*per_iter_full_res;
% for i=2:length(obj_warm_distr)
%     comm_cost_iter_full_res(i) = comm_cost_iter_full_res(i-1) + per_iter_full_res;
%     comm_cost_iter_distr(i) = comm_cost_iter_distr(i-1) + per_iter_distr;
% end

% optimal_line = cost_optimal*ones(1, length(comm_cost_iter_distr));

% h_full_res = plot(comm_cost_iter_full_res, obj_warm_full_res, '-r', 'Linewidth', 2); hold on
% h_distr = plot(comm_cost_iter_distr, obj_warm_distr, '--r', 'Linewidth', 2); hold on
% h_opt = plot(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_distr) ), optimal_line, '-b', 'linewidth', 2); hold on;
% h_opt1 = loglog(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_distr)), optimal_line*1.5, '-c', 'linewidth', 1); hold on;
% h_opt2 = loglog(linspace(min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_full_res, [], 'all'), length(comm_cost_iter_distr)), optimal_line*2, '-c', 'linewidth', 1); hold on;


% legend_text{2} = 'CuteMaxVar, 3 bits'
% legend_text{1} = 'AltMaxVar'
% legend_text{3} = 'optimal'

% t1 = text(comm_cost_iter_full_res(1500), cost_optimal, 'optimal', 'fontsize', 16, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');
% t2 = text(comm_cost_iter_full_res(1500), 1.5*cost_optimal, '1.5 x optimal, CR = 0.9064', 'fontsize', 16, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');
% t3 = text(comm_cost_iter_full_res(1500), 2*cost_optimal, '2 x optimal, CR = 0.9061', 'fontsize', 16, 'HorizontalAlignment','center','VerticalAlignment', 'bottom');

% leg = legend([h_full_res h_distr h_opt], legend_text, 'fontsize', 16);

% xlabel('communication cost (BPV)','fontsize',16)
% ylabel('objective value','fontsize',16)
% % print('-depsc','lambda_1')
% xlim([min(comm_cost_iter_distr, [], 'all') max(comm_cost_iter_full_res, [], 'all')])

% % xlim([min(comm_cost_iter_distr, [], 'all'), max(comm_cost_iter_distr, [], 'all')]);
% ylim([0.15 0.8]);
% ax = gca;
% ax.FontSize=16

% set(gcf, 'PaperPosition', [0 0 8 6]); %Position plot at left hand corner with width 5 and height 5.
% set(gcf, 'PaperSize', [8 6]); %Set the paper to have width 5 and height 5.
% saveas(gcf, '../data/simulation_results/real_optimal_compare', 'pdf') %Save figure