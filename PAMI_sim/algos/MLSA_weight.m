function [ G,Q,cost_MLSA,Li ] = MLSA_weight( X,K,m,norm_vec,vec_ind )
% The method in Multiview LSA: Representation Learning via Generalized CCA
% P. Rastogi, B Van Durme R Arora
r = 1;
[~,J]=size(X);
M = [];

r = 1;

sqrt_norm_vec=sqrt(norm_vec);

for j=1:J
    disp(['runing at ',num2str(j),' th SVD of MLSA'])
    [A{j},S{j},B{j}]=svds(X{j},m);
    Li{j} = (S{j}(1,1))^2;
    s_diag{j} = sqrt(diag(S{j}));
    T{j}=  diag(s_diag{j})*diag(1./sqrt(r+(s_diag{j}).^2)) ;
    M = [M,A{j}*T{j}];  % this dimension is m*J, which is acceptable - or you can try incremental SVD
    
end

M = bsxfun(@rdivide,M,sqrt_norm_vec);



% Compute the svd

[G,~,~]=svd(M,0);

G = G(:,1:K); % this is G_tilde

G = bsxfun(@rdivide,G,sqrt_norm_vec);

cost_MLSA = 0;
for j=1:J
    Gi =  bsxfun(@times,G,vec_ind(:,j));
    %     Q{j}= (X{j}'*X{j}+r*eye(MM))\(X{j}'*G);
    Q{j} = B{j}*diag(1./(r+s_diag{j}.^4))*S{j}*A{j}'*Gi;         % I can certainly keep this 
    cost_MLSA = (1/2)*sum(sum((X{j}*Q{j}-Gi).^2))+cost_MLSA;
end

end

