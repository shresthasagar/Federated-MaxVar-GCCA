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