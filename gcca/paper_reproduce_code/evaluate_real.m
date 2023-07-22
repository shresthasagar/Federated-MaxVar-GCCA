addpath /scratch/sagar/Projects/matlab/lib/tensor_toolbox
addpath ../src

distr = load('../data/simulation_conditions/real_data_distr_sgd_optimal.mat');
full_res = load('../data/simulation_conditions/real_data_full_res_sgd_optimal.mat');
optimal = load('../data/simulation_conditions/real_optimal_altmaxvar.mat');


data = load('../../real_data/Europarl_all.mat');
[~, I] = size(data.data);
I = 3;

for i=1:I 
    X{i} = data.data{i};
end

[J N] = size(X{1});

test_size = 1000;
samp = randsample(1:J, test_size);

for i=1:I 
    X{i} = X{i}(samp,:);
end

[~, max_iter, ~,~] = size(full_res.Q1)

for r=1:max_iter
    for i=1:I
        Q_full{i} = squeeze(full_res.Q1(i,r,:,:));
        Q_distr{i} = squeeze(distr.Q1(i,r,:,:));
    end
    r
    [aroc_full_res(r) nn_freq_full(r)] = eval_europarl(X, Q_full);
    [aroc_distr(r) nn_freq_distr(r)] = eval_europarl(X, Q_distr);
end