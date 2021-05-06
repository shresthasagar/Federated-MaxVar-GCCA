% feature selective CCA


% addpath ../algos

% addpath ../cg_matlab

clear;
clc;
close all


I = 3;
Mr = 30; Mp = 70;
M = Mr + Mp;
L = 1000;
N = 10;

E = randn(N,L);
X = cell(1,I);
Y = cell(1,I);
for i = 1:I
    X{i} = randn(Mr,N) * ( E + 0*randn(N,L) ) + 0*randn(Mr,L);
    Y{i} = [ X{i}; randn(Mp,L) ]';
    Y{i} = Y{i} .* ( rand(size(Y{i})) < .5 );
    X{i} = Y{i}(:,1:Mr);
end


%% global solution
K = N;
ZZ = zeros(L,L);
MM = zeros(L,L); r = .1; M_clean=zeros(L,L);
Ux = cell(I,1); sigx = cell(I,1);
Li = cell(I,1);
for i=1:I
%         M_clean = M_clean + X_clean{i}*((X_clean{i}'*X_clean{i}+r*eye(M))\X_clean{i}')
    MM = MM + L*Y{i}*(((1/L)*Y{i}'*Y{i}+r*eye(M))\Y{i}');
    [Ux{i},sigx{i},~]= svds(Y{i},M);
    ZZ=ZZ+Ux{i}*diag(1./diag(sigx{i}.^2+r*eye(M)))*Ux{i}';  % interesting question: why is this important?
    Li{i} = ((sigx{i}(1,1))^2)/L;
end
[Um,Sm,Vm]=svd(MM,0); 
DiagSm = Sm/I; 


Ubeta = Um(:,K+1:end);
[valMM,order_ind]=sort(diag(Sm),'descend');
G0 = Um(:,order_ind(1:K));%*Vm(:,1:K)';

%  question is: why do I care about unique eigen decomposition?

cost_global = 0;
Q0 = cell(I,1);
for i=1:I 
    Q0{i}= (1/sqrt(L))*(((1/L)*Y{i}'*Y{i}+r*eye(M))\(Y{i}'*G0));
    cost_global= (1/2)*sum(sum(((1/sqrt(L))*Y{i}*Q0{i}-G0).^2))+cost_global+(r/2)*sum(sum(Q0{i}.^2));
end
% global_cost(trial)=cost_global;


%% clean solution
K = N;
ZZ = zeros(L,L);
MM = zeros(L,L); r = .1; M_clean=zeros(L,L);
Ux = cell(I,1); sigx = cell(I,1);
Li = cell(I,1);
for i=1:I
%         M_clean = M_clean + X_clean{i}*((X_clean{i}'*X_clean{i}+r*eye(M))\X_clean{i}')
    MM = MM + L*X{i}*(((1/L)*X{i}'*X{i}+r*eye(Mr))\X{i}');
    [Ux{i},sigx{i},~]= svds(X{i},Mr);
    ZZ=ZZ+Ux{i}*diag(1./diag(sigx{i}.^2+r*eye(Mr)))*Ux{i}';  % interesting question: why is this important?
    Li{i} = ((sigx{i}(1,1))^2)/L;
end
[Um,Sm,Vm]=svd(MM,0); 
DiagSm = Sm/I; 


Ubeta = Um(:,K+1:end);
[valMM,order_ind]=sort(diag(Sm),'descend');
Gc = Um(:,order_ind(1:K));%*Vm(:,1:K)';

%  question is: why do I care about unique eigen decomposition?

cost_global = 0;
Qc = cell(I,1);
for i=1:I 
    Qc{i}= (1/sqrt(L))*(((1/L)*X{i}'*X{i}+r*eye(Mr))\(X{i}'*Gc));
    cost_global= (1/2)*sum(sum(((1/sqrt(L))*X{i}*Qc{i}-Gc).^2))+cost_global+(r/2)*sum(sum(Qc{i}.^2));
end


    
%% this paper
r = 0.1;
K = N; m = N;
MaxIt = 1000;
lambda = .05*ones(1,I);



% initialization
% [ G_ini,Q_ini,Ux,Us,UB,cost_MLSA,Li ] = MLSA( Y,K,m,r);

% plain GCCA
% [ Q0, G0 ,obj0,dist0, St0] = LargeGCCA_new( Y, K, ...
%     'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'MaxIt',MaxIt,'Li',Li, ...
%     'Reg_Type', 'fro', 'lambda', lambda );
% figure
% plot(obj0)

% fresCCA
[ Q1, G1 ,obj1,dist1, St1] = LargeGCCA_new( Y, K, ...
    'G_ini',G0,'Q_ini',Q0,'r',r,'MaxIt',MaxIt,'Li',Li, ...
    'Reg_Type', 'L21', 'lambda', lambda );
figure
plot(obj1)


fprintf('========================================================\n')
disp('plain CCA')
disp('row sparsity:')
figure
for i = 1:I
    sum( sqrt(sum(Q0{i}.^2,2)) < 1e-3 )
    subplot(3,1,i)
    stem( sqrt(sum(Q0{i}.^2,2)), 'o' )
    axis([0,101,0,.3])
end
disp('RMSE')
rmse = 0;
for i = 1:I
    rmse = (1/2)*sum(sum(((1/sqrt(L))*Y{i}*Q0{i}-G0).^2)) + rmse;
end
rmse
disp('RMSE correct feature')
T = 0;
for i = 1:I
    T = T + X{i}*Q0{i}(1:Mr,:);
end
[ Tu,~,Tv ] = svd(T,0); GG0 = Tu*Tv';
rmse = 0;
for i = 1:I
    rmse = (1/2)*sum(sum(((1/sqrt(L))*X{i}*Q0{i}(1:Mr,:)-GG0).^2)) + rmse;
end
rmse

disp('row sparse CCA')
disp('row sparsity:')
figure
for i = 1:I
    sum( sqrt(sum(Q1{i}.^2,2)) < 1e-3 )
    subplot(3,1,i)
    stem( sqrt(sum(Q1{i}.^2,2)), 'o' )
    axis([0,101,0,.3])
end
disp('RMSE')
rmse = 0;
for i = 1:I
    rmse = (1/2)*sum(sum(((1/sqrt(L))*Y{i}*Q1{i}-G1).^2)) + rmse;
end
rmse
disp('RMSE correct feature')
T = 0;
for i = 1:I
    T = T + X{i}*Q1{i}(1:Mr,:);
end
[ Tu,~,Tv ] = svd(T,0); GG1 = Tu*Tv';
rmse = 0;
for i = 1:I
    rmse = (1/2)*sum(sum(((1/sqrt(L))*X{i}*Q1{i}(1:Mr,:)-GG1).^2)) + rmse;
end
rmse

disp('clean data')
rmse = 0;
for i = 1:I
    rmse = (1/2)*sum(sum(((1/sqrt(L))*X{i}*Qc{i}(1:Mr,:)-Gc).^2)) + rmse;
end
rmse





