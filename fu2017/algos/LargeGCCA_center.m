function [ Q,G ,obj,St] = LargeGCCA_center( X,MaxIt,G,Q,Li,EXTRA,WZW,norm_vec,vec_ind )


% here I incorperate centering


[~,I]=size(X);

[L,M]=size(X{1});

[~,K]=size(Q{1});

r = 0;
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
    
    disp(['at iteration ',num2str(it)])

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
                Q{i}=Q_tilde{i}-(1/Li{i})*(X{i}'*CWCXQ{i}-X{i}'*Ci_G{i}+r*Q_tilde{i});
       
            else
                XQ{i} = X{i}*Q{i};
                CXQ{i}= XQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(XQ{i});
                WCXQ{i} = bsxfun(@times,CXQ{i},vec_ind(:,i));
                CWCXQ{i}= WCXQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(WCXQ{i});
                Q{i}=Q{i}-(1/Li{i})*(X{i}'*CWCXQ{i}-X{i}'*Ci_G{i}+r*Q{i});
            end
        else
            XQ{i} = X{i}*Q{i};
            CXQ{i}= XQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(XQ{i});
            WCXQ{i} = bsxfun(@times,CXQ{i},vec_ind(:,i));
            CWCXQ{i}= WCXQ{i} - ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(WCXQ{i});
            Q{i}=Q{i}-(1/Li{i})*(X{i}'*CWCXQ{i}-X{i}'*Ci_G{i}+r*Q{i});
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
        

    
    obj_temp = 0;Q_norm = 0;
    G = bsxfun(@rdivide,G,sqrt_norm_vec);
    for i=1:I
        Gi{i} = bsxfun(@times,G,vec_ind(:,i));
        Ci_G{i} = Gi{i}-ones(L,1)*(1/nnz(vec_ind(:,i)))*sum(Gi{i});
        obj_temp = sum(sum((WCXQ{i} - Gi{i}).^2));
        Q_norm = Q_norm + sum(sum(Q{i}.^2));
    end
    obj(it)=sum(obj_temp)+r*sum(Q_norm);
    fit(it)=sum(obj_temp);
    
    disp(['costvalue is ',num2str(obj(it)),' fitvalue is ',num2str(fit(it))])
    
%    if mod(it,10)==0 % save the result every 50 iterations maybe
%        filename =  ['iteration_center_',num2str(it)];
%        save(filename,'Q','G','obj')
%    end

   one_it_time = toc
     
end 

% timeII = toc
% 
% figure(1)
% semilogy([2:it],obj2(2:end),'-b')

end

