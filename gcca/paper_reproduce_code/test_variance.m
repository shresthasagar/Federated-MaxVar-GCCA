
TotalTrial = 1;
mu = .1;
noise_var = 0.1;
MaxIt = 500;
InnerIt = 10;
Nbits = 3;
r = 0;

I = 3;
L = 5000;
M = 2000;
N = 200;
K = 5;
m = 10;
Z = sprandn(L, N, 1e-3);%*diag(2+1*randn(N,1));
% Z = randn(L, N);%*diag(2+1*randn(N,1));

for i=1:I
    A{i}=sprandn(N, M, 1e-3);            
    X{i}=Z*A{i} + noise_var*sprandn(L, M, 1e-3);
end


% cond_mean(trial)=mean( condnum(i));
tic;
filename = '../data/rand_1.mat';
[ G_ini,Q_ini,Ux,Us,UB,cost_mlsa,Li ] = MLSA( X,K,m,r);
% dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
save(filename,'X','G_ini','Q_ini','Li');

time_mlsa =toc;

init_vars  = load(filename);
X = init_vars.X;
G_ini = init_vars.G_ini;
Q_ini = init_vars.Q_ini;
% cost_MLSA = init_vars.cost_MLSA;
Li = init_vars.Li;


batch_size = 1000;

ids = randsample(1:L, batch_size);
batch= sparse(X{1}(ids,:)); 
G_batch = G_ini(ids,:); 

% current_lr = 1/it;
Q = Q_ini;
gd = ((1/batch_size)*X{1}'*(X{1}*Q{1})+r*Q{1}-(1/sqrt(L))*X{1}'*G_ini);

for n=1:100
    ids = randsample(1:L, batch_size);
    batch= sparse(X{1}(ids,:)); 
    G_batch = G_ini(ids,:); 
    sgd = (L/batch_size)*((1/batch_size)*batch'*(batch*Q{1})+r*Q{1}-(1/sqrt(batch_size))*batch'*G_batch);
    sgd_norm(n) = norm(sgd-gd, 'fro');    
end
mean(sgd_norm)
