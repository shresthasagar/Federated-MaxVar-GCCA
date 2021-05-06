function [ G,Q,A,S,B,cost_MLSA,Li ] = MLSA( X,K,m,r )
% The method in Multiview LSA: Representation Learning via Generalized CCA
% P. Rastogi, B Van Durme R Arora

[~,J]=size(X);

[L,M]=size(X{1});

M = [];

% r = 1;
% load svd_M100_I4

for j=1:J
    disp(['runing at ',num2str(j),' th SVD of MLSA'])
    [A{j},S{j},B{j}]=svds(X{j},m);
    Li{j} = ((S{j}(1,1))^2)/L;
    s_diag{j} = sqrt(diag(S{j}));
    %     SS{j}=S{j}'*diag(1./diag(r*eye(m)+S{j}*S{j}'))*S{j}*A{j}';
    T{j}=  diag(s_diag{j})*diag(1./sqrt(r+(s_diag{j}).^2)) ;
    M = [M,A{j}*T{j}];  % this dimension is m*J, which is acceptable - or you can try incremental SVD
    
end

MM = size(X{1},2);


% Compute the svd

[G,~,~]=svd(M,0);

G = G(:,1:K);

cost_MLSA = 0;
for j=1:J
    %     Q{j}= (X{j}'*X{j}+r*eye(MM))\(X{j}'*G);
    Q{j} = sqrt(L)*(B{j}*diag(1./(r+s_diag{j}.^4))*S{j}*(A{j}'*G));
    cost_MLSA =(1/2)*sum(sum(((1/sqrt(L))*X{j}*Q{j}-G).^2))+cost_MLSA + (r/2)*sum(sum(Q{j}.^2));
end
 
end

