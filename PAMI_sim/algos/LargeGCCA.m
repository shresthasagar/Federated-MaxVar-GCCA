function [ Q,G ,obj,St] = LargeGCCA( X,MaxIt,G,Q,Li,EXTRA,WZW,norm_vec,vec_ind )

[~,I]=size(X);

[L,M]=size(X{1});

[~,K]=size(Q{1});

r = 0;
for i=1:I
    Li{i}=Li{i}+r;
end

tk=1;
Q_old = Q;
Q_tilde = cell(1,I);
for it=1:MaxIt
    
%     if it>500 
%         EXTRA = 0;
%     end
    
    disp(['at iteration ',num2str(it)])

%     tic
    if it>1
    tk_plus_one = (1+sqrt(1+4*tk^2))/2;
    end
    for i=1:I
        %         Q{i}= (X{i}'*X{i})\(X{i}'*G);
        
        if EXTRA ==1
            if it>1
                
                Q_tilde{i} = Q{i} + (tk-1)/tk_plus_one*(Q{i}-Q_old{i});
                
                Q_old{i}=Q{i};
                %                 Q{i}=Q_tilde{i}-(1/Li{i})*(X{i}'*(X{i}*Q_tilde{i})-X{i}'*G);
                Q{i} = Q_tilde{i}-(1/Li{i})*(X{i}'*(X{i}*Q_tilde{i})+r*Q_tilde{i}-X{i}'*G);
            else
                Q{i}=Q{i}-(1/Li{i})*(X{i}'*(X{i}*Q{i})+r*Q{i}-X{i}'*G);
            end
        else
            %             Q{i}=Q{i}-(1/Li{i})*(X{i}'*(X{i}*Q{i})-X{i}'*G); % not
            %             regularized version
           
            Q{i}=Q{i}-(1/Li{i})*(X{i}'*(X{i}*Q{i})+r*Q{i}-X{i}'*G);
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
%     M_temp = bsxfun(@rdivide,M_temp,norm_vec);
    if WZW ==0
        [Ut,St,Vt]=svd(M_temp,0);
        G = Ut(:,1:K)*Vt';
        
    elseif WZW == 1
        % WZW version - for fun
        G0=G;
        opts.record = 0;
        opts.mxitr  = 10;
        opts.gtol = 1e-5;
        opts.xtol = 1e-5;
        opts.ftol = 1e-8;
        opts.tau = 1e-3;St = [];
        [G, out]= OptManiMulitBallGBB(G0, @grad_G, opts, M_temp/I);
    end
    
    obj_temp = 0;
    for i=1:I
%         Gi = bsxfun(@times,G,vec_ind(:,i));
        obj_temp = sum(sum((XQ{i} - G).^2));
    end
    obj(it)=sum(obj_temp);
    
%     disp(['costvalue is ',num2str(obj(it))])
%     
%    if mod(it,10)==0 % save the result every 50 iterations maybe
%        filename =  ['iteration_noextra_',num2str(it)];
%        save(filename,'Q','G','obj')
%    end

%     if it>1&&abs(obj(it)-obj(it-1))<1e-16
%         break;
%     end

%    one_it_time = toc
     
end 
 
% timeII = toc
% 
% figure(1)
% semilogy([2:it],obj2(2:end),'-b')

end

