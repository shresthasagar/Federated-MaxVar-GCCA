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
for trial = 1:TotalTrial
    disp(['at trial ',num2str(trial)])
    I = 3;
    L = 10000;
    M = 5000; 
    N = M;
    K = 10;
    m = 100;
    sparsity_level = 2e-3;
    
    Z = sprandn(L,N,sparsity_level);
    for i=1:I
        A{i}=sprandn(N,M,.00001);
        X{i}=Z*A{i};
        X{i}=sparse(X{i});
    end
    Zf = full(Z);
    [Uz, ~, ~]  = svd(Zf, 0);
    Ubeta = Uz(:, K+1:end);


    %% How to initialize is another problem...
     
    r = 0;
    tic;
    [ G_ini,Q_ini,Ux,Us,UB,cost_MLSA(trial),Li ] = MLSA( X,K,m,r);
    timeMLSA(trial) =toc;
    
    filename = ['trial_',num2str(trial)];
    save(filename,'X','G_ini','Q_ini','cost_MLSA','Li')
    

    
       
    MaxIt = 200; 
    tic

    % load from file
    filename = ['trial_',num2str(trial)]; 
    init_vars  = load(filename);

    X = init_vars.X;
    G_ini = init_vars.G_ini;
    Q_ini = init_vars.Q_ini;
    cost_MLSA = init_vars.cost_MLSA;
    Li = init_vars.Li;

    %%
    % XX = zeros(L,M);
    % for i=1:i
    %     XX = XX + X{i}*inv(X{i}'*X{i})*X{i}';
    % end
    % [Um, ~,~] = svd(M,0);

    %%
    % [Q,G_1,obj1(trial,:),~,St1] = LargeGCCA_federated( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',10, 'Reg_type', 'none');

    [Q2,G_2,obj2(trial,:),dist2,St2] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',10, 'Reg_type', 'none', 'Um', Ubeta);

    time_proposed_1(trial) = toc;
   
end   
    
%     tic
%     [Q,G_2,obj2(trial,:),~,St2] = LargeGCCA_new( X,K,'G_ini',G_ini,'Q_ini',Q_ini,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',10,'EXTRA',0 );
%     time_proposed_10(trial) = toc;
 

    





     
   

%     tic
%     for i=1:I
%         Q_rnd{i}=randn(M,K);  
%     end
%     G_rnd = randn(L,K);
%     [Q,G_1,obj11(trial,:),dist11(trial,:),St1] = LargeGCCA_new( X,K,'G_ini',G_rnd,'Q_ini',Q_rnd,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1);
%     time_proposed_warm1(trial) = toc;
%     EXTRA = 0;
%     WZW = 0;

 
%     tic

%     [Q,G_2,obj22(trial,:),dist22(trial,:),St2] = LargeGCCA_new( X,K,'G_ini',G_rnd,'Q_ini',Q_rnd,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',10,'EXTRA',0);
%     time_proposed_warm10(trial) = toc;
%     EXTRA = 0;
%     WZW = 0;
    
    
%     for i=1:I
%        Y{i}=(1/sqrt(L))*X{i}; 
%     end
    
%     tic
    
    
%     [G] = my_gcca(Y,K);
%     time_lanc(trial) = toc;
%     cost_lanc = 0;
%     for j=1:I
%          Q{j} = pcg_mat(@(v)XtX(Y{j},v),Y{j}'*G);
%          cost_lanc =(1/2)*sum(sum((Y{j}*Q{j}-G).^2))+cost_lanc;
%     end
    
% %     tic
     
% %     tic
% %     
% %     [Q,G_2,obj33(trial,:),dist33(trial,:),St3] = LargeGCCA_new( X,K,'G_ini',G_rnd,'Q_ini',Q_rnd,'r',r,'algo_type','plain','Li',Li,'MaxIt',MaxIt,'Inner_it',1000,'EXTRA',1,'UM',Ubeta );
% %     time_proposed = toc;
% %     EXTRA = 0;
% %     WZW = 0;

    
%     % how to measure?
% %     
% %     for i=1:L
% %         for j=1:L
% %             T(i,j)=norm(Z(i,:)-Z(j,:),2);
% %             O(i,j)=norm(G(i,:)-G(j,:),2);
% %         end
% %     end
% %     
% % 
% %     score_LGCCA = trace(T'*O)/(sqrt(trace(T'*T))*sqrt(trace(O'*O)));
% %     
% end  
  
 

% figure(1)
% it1 = length(obj1(1,:));
% loglog([1:it1],mean(obj1(:,1:end)),'-r','linewidth',2);hold on

% it2 = length(obj2(1,:));
% loglog([1:it2],mean(obj2(:,1:end)),'-b','linewidth',2); hold on


% % it3 = length(obj3(1,:));
% % loglog([1:it3],mean(obj3(:,1:end)),'-m','linewidth',2); hold on

% it1 = length(obj11(1,:));
% loglog([1:it1],mean(obj11(:,1:end)),'--r','linewidth',2);hold on

% it2 = length(obj22(1,:));
% loglog([1:it2],mean(obj22(:,1:end)),'--b','linewidth',2); hold on

% % it2 = length(obj33(1,:));
% % loglog([1:it2],mean(obj33(:,1:end)),'--m','linewidth',2); hold on


% % obj4=ones(1,max(it1,it2))*mean(global_cost);
% % loglog([1:max(it1,it2)],obj4(1:end),'-k','linewidth',2); hold on

% obj5=ones(1,max(it1,it2))*mean(cost_MLSA);


% loglog([1:max(it1,it2)],obj5(1:end),'--k','linewidth',2); hold on
% legend('Proposed (T=1, warm)','Proposed (T = 10, warm)','Proposed (T = 1,randn)','Proposed (T=10,randn)','MVLSA')
% % title('X_i: 50K x 10K matrix; sparsity = 0.001; nnz = 501,111; Comp. = 50')
% set(gca,'fontsize',14)
% xlabel('iterations','fontsize',14)
% ylabel('cost value','fontsize',14)


% % obj1(1,end)
% % obj2(1,end)
% % global_cost

% xlim([1 MaxIt])
% print('-depsc','cost_large')

% save may7_download time_proposed_warm1 time_proposed_warm10 timeMLSA obj1 obj2 obj11 obj22 cost_MLSA MaxIt


%%
% figure(2)
% it1 = length(obj1(1,:));
% semilogy([2:it1],mean(dist1(:,2:end)),'-r','linewidth',2);hold on
% 
% it2 = length(dist2(1,:));
% semilogy([2:it2],mean(dist2(:,2:end)),'-b','linewidth',2); hold on
% 
% % it2 = length(dist2(1,:));
% % semilogy([2:it2],mean(dist3(:,2:end)),'-m','linewidth',2); hold on
% 
% it1 = length(dist11(1,:));
% semilogy([2:it1],mean(dist11(:,2:end)),'--r','linewidth',2);hold on
% 
% it2 = length(dist22(1,:));
% semilogy([2:it2],mean(dist22(:,2:end)),'--b','linewidth',2); hold on
% % 
% % it2 = length(dist22(1,:));
% % semilogy([2:it2],mean(dist33(:,2:end)),'--m','linewidth',2); hold on
% 
% legend('Proposed (T=1, warm)','Proposed (T=10, warm)','Proposed (T = 1,randn)','Proposed (T=10, randn)')
% % title('X_i: 50K x 10K matrix; sparsity = 0.001; nnz = 501,111; Comp. = 50')
% set(gca,'fontsize',14)
% xlabel('iterations','fontsize',14)
% ylabel('dist','fontsize',14)
% % print('-depsc','sanity_MLSA_ini')
% 
% % xlim([1 1000])
% % ylim([0 20])
% 
% print('-depsc','dist_large')







