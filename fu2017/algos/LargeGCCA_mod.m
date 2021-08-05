function [ Q,G ,obj,St] = LargeGCCA_mod( X,MaxIt,G,Q,Li,EXTRA,WZW,norm_vec,vec_ind )

[~,I]=size(X);

[L,M]=size(X{1});

[~,K]=size(Q{1});

r = 1;
for i=1:I
    Gi{i} = bsxfun(@times,G,vec_ind(:,i));
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
        %         Q{i}= (X{i}'*X{i})\(X{i}'*G);
        
        if EXTRA ==1
            if it>1
               
                Q_tilde{i} = Q{i} + (tk-1)/tk_plus_one*(Q{i}-Q_old{i});
                
                Q_old{i}=Q{i};
                Q{i} = Q_tilde{i}-(1/Li{i})*(X{i}'*(X{i}*Q_tilde{i})+r*Q_tilde{i}-X{i}'*Gi{i} );
            else
                Q{i}=Q{i}-(1/Li{i})*(X{i}'*(X{i}*Q{i})+r*Q{i}-X{i}'*Gi{i} );
            end
        else
            Q{i}=Q{i}-(1/Li{i})*(X{i}'*(X{i}*Q{i})+r*Q{i}-X{i}'*Gi{i} );
        end
    end
    if it>1
        tk = tk_plus_one;
    end
    M_temp = zeros(L,K); 
    for i=1:I
        XQ{i}=X{i}*Q{i};
        M_temp = M_temp + XQ{i};
    end
    
    
    % SVD version - global optimality guaranteed
    M_temp = bsxfun(@rdivide,M_temp,sqrt_norm_vec);
  
    [Ut,St,Vt]=svd(M_temp/I,0);
    G = Ut(:,1:K)*Vt';
        

    
    obj_temp = 0;
    G = bsxfun(@rdivide,G,sqrt_norm_vec);
    for i=1:I
        Gi{i}  = bsxfun(@times,G,vec_ind(:,i));
        obj_temp = sum(sum((XQ{i} - Gi{i}).^2));
    end
    obj(it)=sum(obj_temp);
    
    disp(['costvalue is ',num2str(obj(it))])
%     
%    if mod(it,10)==0 % save the result every 50 iterations maybe
%        filename =  ['iteration_noCentering_',num2str(it+2000)];
%        save(filename,'Q','G','obj')
%    end

   one_it_time = toc
     
end 

% timeII = toc
% 
% figure(1)
% semilogy([2:it],obj2(2:end),'-b')

end

