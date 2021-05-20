% A simple script to show that local is global
% min_{G Q_i} sum_i||X_i*Q_i-G||
% st G^tG = I;
% X_i is tall and full rank

addpath ../algos

addpath ../cg_matlab

clear;
clc;
close;


 
TotalTrial = 1;
dev = .1;
MaxIt = 1000
for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])
    % dense: sanity check
    I = 3;
    L = 500;
    Lbad = 0;
    M = 100;
    N = 10;
    K = 5;
    m = 50;
    Z = randn(L,N);%*diag(2+1*randn(N,1));
    for i=1:I
        A{i}=randn(N,M);%+ 1*eye(N,N);
%         [Ua,~,Va]=svd(A{i});
%         A{i}=Va(1:N,:)+ dev*randn(N,M);
        
        X{i}=Z*A{i} + .1*randn(L,M); 
        condnum(i)=cond((1/L)*X{1}'*X{1});
        
        % X{i}(:,end-Lbad+1:end)=randn(L,Lbad);
    end
    
    
    
    cond_mean(trial)=mean( condnum(i));
    
    %% computing the global solution
    tic
    ZZ = zeros(L,L);
    MM = zeros(L,L); r = .1; M_clean=zeros(L,L);
    for i=1:I
%         M_clean = M_clean + X_clean{i}*((X_clean{i}'*X_clean{i}+r*eye(M))\X_clean{i}')
        MM = MM + L*X{i}*(((1/L)*X{i}'*X{i}+r*eye(M))\X{i}');
        [Ux{i},sigx{i},~]= svd(X{i});
        ZZ=ZZ+Ux{i}(:,1:M)*inv(sigx{i}(1:M,:).^2+r*eye(M))*Ux{i}(:,1:M)';  % interesting question: why is this important?
    end
    [Um,Sm,Vm]=svd(MM); 
    DiagSm = Sm/I; 

    % Ubeta = Um(:,1:K);

    Ubeta = Um(:,K+1:end);
    tic;

    [ G_ini,Q_ini,Ux,Us,UB,cost_MLSA(trial),Li ] = MLSA( X,K,m,r);
    dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
    timeMLSA =toc;

    %%  random initialization
    tic
    for i=1:I
        Q_ini{i}=randn(M,K);   
    end
    G_ini = randn(L,K);
    
    [Q,G_1,obj1(trial,:),dist1,St1] = LargeGCCA_federated( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',100, 'Reg_type', 'fro', 'Um', Ubeta);
    
    [Q2,G_2,obj2(trial,:),dist2,St2] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',100, 'Reg_type', 'fro', 'Um', Ubeta);

    
end