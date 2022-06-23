%% Visualization of Communication cost (bytes) vs Iteration
figure(1)

TotalTrial = 50;
mu = .1;
noise_var = 0.1;
MaxIt = 1000;
InnerIt = 10;
Nbits = 3;
r = 0;
batch_size = 150;
shaded_area_alpha = 0.3;


bits_list = [2, 3, 4, 5]


filename = '../data/simulation_conditions/small_bits.mat';
small = load(filename);
obj_warm_full_res = small.obj_warm_full_res;
obj_warm_distr = small.obj_warm_distr;
t_warm_full_res = small.t_warm_full_res;
t_warm_distr = small.t_warm_distr;
cost_optimal = small.cost_optimal;

std_full_res = squeeze(std(obj_warm_full_res, 1));
std_distr = squeeze(std(obj_warm_distr, 1));
m1 = squeeze(mean(obj_warm_full_res, 1));
m2 = squeeze(mean(obj_warm_distr, 1));
area_full_res = [m1'+std_full_res', fliplr(m1'-std_full_res')];
area_distr = [m2 + std_distr, flip(m2 - std_distr, 2)];

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
x_full_res = [comm_cost_iter_full_res, fliplr(comm_cost_iter_full_res)];
h_area = fill(x_full_res, area_full_res, colors_full_res{length(bits_list)}, 'linestyle', 'none');
set(h_area, 'facealpha', shaded_area_alpha);
legend_text_full_res = ['AltMaxVar'];

for idx=1:length(bits_list)
    h_distr(idx) = loglog(comm_cost_iter_distr(idx,:), squeeze(mean(obj_warm_distr(:,idx,:),1)), colors_distr{idx}, 'Linewidth', 2); hold on;
    x_distr = [comm_cost_iter_distr(idx,:), fliplr(comm_cost_iter_distr(idx,:))];
    h_area_distr = fill(x_distr, area_distr(idx,:), colors_distr{idx}, 'linestyle', 'none');
    set(h_area_distr, 'facealpha', shaded_area_alpha);
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
x_full_res = [[1:MaxIt+1], fliplr([1:MaxIt+1])];
h_area = fill(x_full_res, area_full_res, colors_full_res{length(bits_list)}, 'linestyle', 'none');
set(h_area, 'facealpha', shaded_area_alpha);
legend_text_full_res = ['AltMaxVar, Full gradient']; 

for idx=1:length(bits_list)
    h_distr(idx) = loglog([1:MaxIt+1], squeeze(mean(obj_warm_distr(:,idx,:), 1)), colors_distr{idx}, 'Linewidth', 2); hold on
    x_distr = [[1:MaxIt+1], fliplr([1:MaxIt+1])];
    h_area_distr = fill(x_distr, area_distr(idx,:), colors_distr{idx}, 'linestyle', 'none');
    set(h_area_distr, 'facealpha', shaded_area_alpha);
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

