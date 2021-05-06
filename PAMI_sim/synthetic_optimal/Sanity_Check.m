% A simple script to show that local is global
% min_{G Q_i} sum_i||X_i*Q_i-G||
% st G^tG = I;
% X_i is tall and full rank

addpath /home/xfu/simulations/PAMI_sim/algos

addpath /home/xfu/simulations/PAMI_sim/cg_matlab

clear;
clc;
close;


 
TotalTrial = 2;
dev = .1;
for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])
    % dense: sanity check
    I = 3;
    L = 500;
    Lbad = 50;
    M = 200;
    N = 100;
    K = 5;
    m = 50;
    Z = randn(L,N);%*diag(2+1*randn(N,1));
    for i=1:I
        A{i}=randn(N,M);%+ 1*eye(N,N);
%         [Ua,~,Va]=svd(A{i});
%         A{i}=Va(1:N,:)+ dev*randn(N,M);
        
        X{i}=Z*A{i} + .1*randn(L,M); 
        condnum(i)=cond((1/L)*X{1}'*X{1});
        
        X{i}(:,end-Lbad+1:end)=randn(L,Lbad);
    end
    
    
    
    cond_mean(trial)=mean( condnum(i));
    
    %% computing the global solution
%     tic
%     ZZ = zeros(L,L);
%     MM = zeros(L,L); r = .1; M_clean=zeros(L,L);
%     for i=1:I
% %         M_clean = M_clean + X_clean{i}*((X_clean{i}'*X_clean{i}+r*eye(M))\X_clean{i}')
%         MM = MM + L*X{i}*(((1/L)*X{i}'*X{i}+r*eye(M))\X{i}');
%         [Ux{i},sigx{i},~]= svd(X{i});
%         ZZ=ZZ+Ux{i}(:,1:M)*inv(sigx{i}(1:M,:).^2+r*eye(M))*Ux{i}(:,1:M)';  % interesting question: why is this important?
%     end
%     [Um,Sm,Vm]=svd(MM,0); 
%     DiagSm = Sm/I; 
%      
%     
%     Ubeta = Um(:,K+1:end);
%     [valMM,order_ind]=sort(diag(Sm),'descend');
%     G = Um(:,order_ind(1:K));%*Vm(:,1:K)';
     
    %  question is: why do I care about unique eigen decomposition?
%     
%     cost_global = 0;
%     for i=1:I 
%         Q{i}= (1/sqrt(L))*(((1/L)*X{i}'*X{i}+r*eye(M))\(X{i}'*G));
%         cost_global= (1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+cost_global+(r/2)*sum(sum(Q{i}.^2));
%     end
%     global_cost(trial)=cost_global;
%     timeI = toc
    
   
     
    %% How to initialize is another problem...
     
      
%     tic;
%     [ G_ini,Q_ini,Ux,Us,UB,cost_MLSA(trial),Li ] = MLSA( X,K,m,r);
%     dist_MLSA(trial)=norm(G_ini'*Ubeta,2);
%     timeMLSA =toc;
     
     MaxIt = 1000; 
%     tic
%   
%     [Q,G_1,obj1(trial,:),dist1(trial,:),St1] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1,'Reg_type','L21','UM',Ubeta );
%     time_proposed = toc;
%     
%     tic
%     [Q,G_2,obj2(trial,:),dist2(trial,:),St2] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1000,'EXTRA',1,'UM',Ubeta ,'Reg_type','L21');
%     time_proposed = toc;
%     EXTRA = 0;
%     WZW = 0;
%     tic




     
   
%%  random initialization
    r = .01;
    tic
    for i=1:I
        Li{i} = eigs((1/L)*X{i}'*X{i},1,'LM');
        Q_rnd{i}=randn(M,K);   
    end
    G_rnd = randn(L,K);
    [Q,G_1,obj11(trial,:),dist11(trial,:),St1] = LargeGCCA_new( X,K,'G_ini',G_rnd,'Q_ini',Q_rnd,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',10,'EXTRA',1,'Reg_type','L21' );
    time_proposed = toc;
    EXTRA = 0;
    WZW = 0;
    q_norm = sqrt(sum(Q{1}.^2,2));
    figure(11)
    subplot(2,1,1)
    stem(q_norm)
  
    tic
    [Q,G_2,obj22(trial,:),dist22(trial,:),St2] = LargeGCCA_new( X,K,'G_ini',G_rnd,'Q_ini',Q_rnd,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',10,'EXTRA',1,'Reg_type','L21' );
    time_proposed = toc;
    EXTRA = 0;
    WZW = 0;
    
    q_norm = sqrt(sum(Q{1}.^2,2));
    figure(11)
    subplot(2,1,2)
    stem(q_norm)
  
end  

% St1
% St2
% DiagSm(1:K,1:K)
% 
figure(1)
% it1 = length(obj1(1,:));
% loglog([1:it1],mean(obj1(:,1:end)),'-r','linewidth',2);hold on
% 
% it2 = length(obj2(1,:));
% loglog([1:it2],mean(obj2(:,1:end)),'-b','linewidth',2); hold on
% 
% 
it1 = length(obj11(1,:));
loglog([1:it1],mean(obj11(:,1:end)),'--r','linewidth',2);hold on
% 
it2 = length(obj22(1,:));
loglog([1:it2],mean(obj22(:,1:end)),'--b','linewidth',2); hold on
% 
% 
% obj3=ones(1,max(it1,it2))*mean(global_cost);
% loglog([1:max(it1,it2)],obj3(1:end),'-k','linewidth',2); hold on
% 
% obj4=ones(1,max(it1,it2))*mean(cost_MLSA);
% 
% obj5=ones(1,max(it1,it2))*mean(cost_lanc);
% 
% loglog([1:max(it1,it2)],obj4(1:end),'--k','linewidth',2); hold on
% loglog([1:max(it1,it2)],obj5(1:end),'-g','linewidth',2); hold on
% legend('Proposed (T=1, warm)','Proposed (T = 100, warm)','Proposed (T = 1,randn)','Proposed (T=100,randn)','Global Opt.','MVLSA','CG-Lanczos')
% % title('X_i: 50K x 10K matrix; sparsity = 0.001; nnz = 501,111; Comp. = 50')
% set(gca,'fontsize',14)
% xlabel('iterations','fontsize',14)
% ylabel('cost value','fontsize',14)
% % print('-depsc','sanity_MLSA_ini')
% 
% obj1(1,end)
% obj2(1,end)
% global_cost
% 
% print('-depsc','lambda_1')




%%
% figure(2)
% it1 = length(obj1(1,:));
% semilogy([2:it1],mean(dist1(:,2:end)),'-r','linewidth',2);hold on
% 
% it2 = length(dist2(1,:));
% semilogy([2:it2],mean(dist2(:,2:end)),'-b','linewidth',2); hold on
% 
% 
% it1 = length(dist11(1,:));
% semilogy([2:it1],mean(dist11(:,2:end)),'--r','linewidth',2);hold on
% 
% it2 = length(dist22(1,:));
% semilogy([2:it2],mean(dist22(:,2:end)),'--b','linewidth',2); hold on
% 
% 
% % dist3=ones(1,max(it1,it2))*0;
% % semilogy([2:max(it1,it2)],dist3(2:end),'-k','linewidth',2); hold on
% 
% % dist4=ones(1,max(it1,it2))*mean(dist_MLSA);
% 
% 
% % semilogy([2:max(it1,it2)],dist4(2:end),'--k','linewidth',2); hold on
% legend('Proposed (T=1, warm)','Proposed (solved, warm)','Proposed (T = 1,randn)','Proposed (solved,randn)')
% % title('X_i: 50K x 10K matrix; sparsity = 0.001; nnz = 501,111; Comp. = 50')
% set(gca,'fontsize',14)
% xlabel('iterations','fontsize',14)
% ylabel('dist','fontsize',14)
% % print('-depsc','sanity_MLSA_ini')
% 
% xlim([1 1000])
% ylim([0 20])
% 
% print('-depsc','dist_lambda_1')







