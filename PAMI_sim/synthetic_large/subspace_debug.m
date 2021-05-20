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
MaxIt = 200
for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])
    % dense: sanity check
    I = 3;
    L = 10;
    Lbad = 0;
    M = 5;
    N = 5;
    K = 3;
    m = 4;
    Z = randn(L,N);%*diag(2+1*randn(N,1));
    for i=1:I
        A{i}=randn(N,M);%+ 1*eye(N,N);
        [Ua,~,Va]=svd(A{i});
        A{i}=Va(1:N,:);%+ dev*randn(N,M);
        
        X{i}=Z*A{i};% .1*randn(L,M); 
    end
    
    
    
    % computing the global solution
    tic
    ZZ = zeros(L,L);
    MM = zeros(L,L); r = .1; M_clean=zeros(L,L);
    for i=1:I
%         M_clean = M_clean + X_clean{i}*((X_clean{i}'*X_clean{i}+r*eye(M))\X_clean{i}')
        MM = MM + L*X{i}*(((1/L)*X{i}'*X{i}+r*eye(M))\X{i}');
        [Ux{i},sigx{i},~]= svd(X{i});
        ZZ=ZZ+Ux{i}(:,1:M)*inv(sigx{i}(1:M,:).^2+r*eye(M))*Ux{i}(:,1:M)';  % interesting question: why is this important?
    end
    [Um,Sm,Vm]=svd(MM,0); 
    DiagSm = Sm/I; 
    Ubeta = Um(:,K+1:end);
    tic;
    % Mx = zeros(L,L);
    % for i=1:I
    %     Mx = Mx + (1/sqrt(L))*X{i}*((1/L)*(X{i}'*X{i})\((1/sqrt(L))*X{i}'));
    % end

    % [Ux, Sx, Vx] = svd(Mx, 0);
    % Ubeta = Ux(:,K+1:end);
    % r = 0.1;

    [ G_ini,Q_ini,Ux,Us,UB,cost_MLSA(trial),Li ] = MLSA( X,K,m,r);
    dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
    timeMLSA =toc;

    % G_opt = Ux(:,1:K);
    % for i=1:I
    %     Q_opt{i} = (X{i}'*X{i})\X{i}'*G_opt;
    % end
    % obj_opt = Cost(X, Q_opt, G_opt)

    % subspace_dist_opt = norm(Ubeta'*G_opt, 2)
    % [Uz, Sz, Vz] = svd(Z);
    % Ubeta = Uz(:,K+1:end);

    % G_ini = randn(L,K);
    % for i=1:I
    %     Q_ini{i} = randn(M,K);
    % end

    % [Q,G_1,obj1(trial,:),dist1,St1] = LargeGCCA_federated( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',10, 'Reg_type', 'fro', 'Um', Ubeta);
    
    [U,D] = eig(MM)
    U2 = U(:,1:K);

    [Q2,G_2,obj2(trial,:),dist2,St2] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',10, 'Reg_type', 'none', 'Um', U2);


    
end

function obj = Cost(X,Q,G)
    I = 3;
    L = 10;
    Lbad = 0;
    M = 5;
    N = 5;
    K = 3;
    m = 4;
    obj_temp = 0;
    for i=1:3
        obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+ obj_temp;
    end
    obj = obj_temp;
end