function [ Q,G ,obj,St] = LargeGCCA_center_feature( X,MaxIt,G,Q,Li,r,EXTRA,WZW,norm_vec,vec_ind )


% here I incorperate centering


[~,I]=size(X);

[L,M]=size(X{1});

[Lprime,K]=size(Q{1});
lambda = r*ones(1,Lprime);

for i=1:I
    Gi{i} = bsxfun(@times,G,vec_ind(:,i));
    Ci_G{i} = Gi{i}-ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(Gi{i}); % Ci'*Wi*G in the paper
    Li{i}=Li{i}+r;
end

sqrt_norm_vec=sqrt(norm_vec);


tk=1;
Q_old = Q;
Q_tilde = cell(1,I);
for it=1:MaxIt
    
%     if it>500 
%         EXTRA = 0;
%     end
    
%     disp(['at iteration ',num2str(it)])

    tic
    if it>1
        tk_plus_one = (1+sqrt(1+4*tk^2))/2;
    end
    for i=1:I
        if EXTRA ==1
            if it>1
                Q_tilde{i} = Q{i} + (tk-1)/tk_plus_one*(Q{i}-Q_old{i});
                Q_old{i}=Q{i};
                
                
                
                XQ{i} = X{i}*Q_tilde{i};
                CXQ{i}= XQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(XQ{i});
                WCXQ{i} = bsxfun(@times,CXQ{i},vec_ind(:,i));
                CWCXQ{i}= WCXQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(WCXQ{i});
                
                %                 Q{i} = Q_tilde{i}-(1/Li{i})*(X{i}'*(X{i}*Q_tilde{i})+r*Q_tilde{i}-X{i}'*Ci_G{i});
                Q{i}=Q_tilde{i}-(1/Li{i})*(X{i}'*CWCXQ{i}-X{i}'*Ci_G{i});
                Q{i}  = (shrinkL1Lp(Q{i},(1/Li{i})*lambda,2));
            else
                XQ{i} = X{i}*Q{i};
                CXQ{i}= XQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(XQ{i});
                WCXQ{i} = bsxfun(@times,CXQ{i},vec_ind(:,i));
                CWCXQ{i}= WCXQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(WCXQ{i});
                Q{i}=Q{i}-(1/Li{i})*(X{i}'*CWCXQ{i}-X{i}'*Ci_G{i});
                Q{i}  = (shrinkL1Lp(Q{i},(1/Li{i})*lambda,2));
            end
        else
            XQ{i} = X{i}*Q{i};
            CXQ{i}= XQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(XQ{i});
            WCXQ{i} = bsxfun(@times,CXQ{i},vec_ind(:,i));
            CWCXQ{i}= WCXQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(WCXQ{i});
            Q{i}=   Q{i}-(1/Li{i})*(X{i}'*CWCXQ{i}-X{i}'*Ci_G{i});
            Q{i}  = (shrinkL1Lp(Q{i},(1/Li{i})*lambda,2));
        end
    end
    if it>1
        tk = tk_plus_one;
    end
    M_temp = zeros(L,K); 
    for i=1:I
        XQ{i}=X{i}*Q{i};
        CXQ{i}= XQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(XQ{i});
        WCXQ{i} = bsxfun(@times,CXQ{i},vec_ind(:,i));
        M_temp = M_temp + WCXQ{i};
    end
    
    
    % SVD version - global optimality guaranteed
    M_temp = bsxfun(@rdivide,M_temp,sqrt_norm_vec);
  
    [Ut,St,Vt]=svd(M_temp/I,0);
    G = Ut(:,1:K)*Vt';
        

    
    obj_temp = 0;
    G = bsxfun(@rdivide,G,sqrt_norm_vec);
    for i=1:I
        Gi{i} = bsxfun(@times,G,vec_ind(:,i));
        Ci_G{i} = Gi{i}-ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(Gi{i});
        obj_temp = (1/2)*sum(sum((WCXQ{i} - Gi{i}).^2))+lambda(1)*sum(sqrt(sum(Q{i}.^2,2)));
    end
    obj(it)=sum(obj_temp);
    
    disp(['costvalue is ',num2str(obj(it))])
    
    Sparsity = nnz(sum(Q{1},2))/(size(Q{1},1));
    
    disp(['sparsity of Q1 is ',num2str(Sparsity)]);
    
%    if mod(it,10)==0 % save the result every 50 iterations maybe
%        filename =  ['iteration_feature_',num2str(r),'_',num2str(it)];
%        save(filename,'Q','G','obj')
%    end

%    one_it_time = toc
     
end 

% timeII = toc
% 
% figure(1)
% semilogy([2:it],obj2(2:end),'-b')

end

function C2 = shrinkL1Lp(C1,lambda,p)

% This function solves the shrinkage/thresholding problem for different
% norms p in {1, 2, inf}
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2014
%--------------------------------------------------------------------------

C2 = [];
if ~isempty(lambda)
    [D,N] = size(C1);
    if (p == 1)
        C2 = max(abs(C1)-repmat(lambda,N,1),0) .* sign(C1);
    elseif (p == 2)
        r = zeros(D,1);
        for j = 1:D
            r(j) = max(norm(C1(j,:))-lambda(j),0);
        end
        C2 = repmat(r./(r+lambda'),1,N) .* C1;
    elseif(p == inf)
        C2 = zeros(D,N);
        for j = 1:D
            C2(j,:) =      L2_Linf_shrink(C1(j,:)',lambda(j))';
%               if 2*lambda(j)<1e-10
%                   C2(j,:)= C1(j,:);
%               else
%               C2(j,:) = (C1(j,:)' - ProjectOntoL1Ball(C1(j,:)', 2*lambda(j))).';
%               end
        end
    end
end
end

function x = L2_Linf_shrink(y,t)
% This function minimizes
%     0.5*||b*x-y||_2^2 + t*||x||_inf
% where b is a scalar.  Note that it suffices
% to consider the minimization
%     0.5*||x-y||_2^2 + t/b*||x||_inf
% and so we will assume that the value of b has
% been absorbed into t (= tau).
% The minimization proceeds by initializing
% x with y.  Let z be y re-ordered so that
% the abs(z) is in descending order.  Then
% first solve
%     min_{b>=abs(z2)} 0.5*(b-abs(z1))^2 + t*b
% if b* = abs(z2), then repeat with first and
% second largest z values;
%     min_{b>=abs(z3)} 0.5*(b-abs(z1))^2+0.5*(b-abs(z2))^2 + t*b
% which by expanding the square is equivalent to
%     min_{b>=abs(z3)} 0.5*(b-mean(abs(z1),abs(z2)))^2 + t*b
% and repeat this process if b*=abs(z3), etc.
% This reduces problem to finding a cut-off index, where
% all coordinates are shrunk up to and including that of
% the cut-off index.  The cut-off index is the smallest
% integer k such that
%    1/k sum(abs(z(1)),...,abs(z(k))) - t/k <= abs(z(k+1))
%

x = y;
[dummy,o] = sort(abs(y),'descend');
z = y(o);
mz = abs(z);

% find cut-off index
cs = cumsum(abs(z(1:length(z)-1)))./(1:length(z)-1)'-t./(1:length(z)-1)';
d = (cs>abs(z(2:length(z))));
if sum(d) == 0
    cut_index = length(y); 
else
    cut_index = min(find(d==1));
end

% shrink coordinates 1 to cut_index
zbar = mean(abs(z(1:cut_index)));
if cut_index < length(y)
    x(o(1:cut_index)) = sign(z(1:cut_index))*max(zbar-t/cut_index,abs(z(cut_index+1)));
else
    x(o(1:cut_index)) = sign(z(1:cut_index))*max(zbar-t/cut_index,0);
end

end

