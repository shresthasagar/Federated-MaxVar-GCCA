function [ Q, G ,obj,dist, St] = LargeGCCA_new( X,K, varargin )

    %MaxIt,G,Q,Li,EXTRA,WZW,norm_vec,vec_ind


    if (nargin-length(varargin)) ~= 2
        error('Wrong number of required parameters');
    end

    %--------------------------------------------------------------
    % Set the defaults for the optional parameters
    %--------------------------------------------------------------

    [~,I]=size(X);
    for i=1:I
    [L,M(i)]=size(X{i});
    end
    MaxIt = 1000;
    EXTRA = 0;
    Um = [];
    T = 2;
    L11=0; L21=0;r=0;
    %--------------------------------------------------------------
    % Read the optional parameters
    %--------------------------------------------------------------
    if (rem(length(varargin),2)==1)
        error('Optional parameters should always go by pairs');
    else
        for i=1:2:(length(varargin)-1)
            switch upper(varargin{i})
                case 'R'  % regularization parameter
                    r = varargin{i+1};
                case 'LAMBDA'  % regularization parameter
                    lambda = varargin{i+1};
                case 'MAXIT'
                    MaxIt = varargin{i+1};
                case 'G_INI'
                    G = varargin{i+1};
                case 'Q_INI'
                    Q = varargin{i+1};
                case 'LI'
                    Li = varargin{i+1};
                case 'NORM_VEC' % vector for weighting/ normalization
                    norm_vec = varargin{i+1};
                case 'VEC_IND'
                    vec_ind = varargin{i+1}; % vec_ind(:,i) indicates which row is missing in Xi
                case 'ALGO_TYPE'
                    algo_type = varargin{i+1}; %'plain','centered','plain_fs' (fs: feature-selective)
                case 'INNER_IT'
                    T =  varargin{i+1};
                case 'EXTRA'
                    EXTRA =  varargin{i+1};
                case 'REG_TYPE'
                    REG_TYPE = varargin{i+1}; %'none','fro','L21','L11'
                case 'UM'
                    Um =  varargin{i+1}; % for measuring error
                otherwise
                    % Hmmm, something wrong with the parameter string
                    error(['Unrecognized option: ''' varargin{i} '''']);
            end;
        end;
    end


    switch REG_TYPE
        case 'none'
            L11=0;L21=0;r=0;
        case  'L11'
            L11 =1 ;
        case 'L21'
            L21 = 1;
        case 'fro'
            r = r;
    end



    %%



    [L,M(1)]=size(X{1});

    obj_temp = 0;
    switch REG_TYPE

        case 'fro'
            for i=1:I
                obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+ (r/2)*sum(sum(Q{i}.^2)) + obj_temp;
            end
            obj_0=sum(obj_temp);
        case 'none'
            for i=1:I
                obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+ obj_temp;
            end
            obj_0=sum(obj_temp);
        case 'L21'
            for i=1:I
                obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2)) + (r/2)*sum(sum(Q{i}.^2))+ (lambda)*sum(sqrt(sum(Q{i}.^2,2))) + obj_temp;
            end
            obj_0=sum(obj_temp);
        case 'L11'
            for i=1:I
                obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+ (r/2)*sum(sum(Q{i}.^2)) + (lambda)*sum(sum(abs(Q{i}))) + obj_temp;
            end
            obj_0=sum(obj_temp);
    end
    obj_0=sum(obj_temp);

    if isempty(Um)~=1
        dist_0 = norm(Um'*G,2);
    else dist_0=[];
    end

    for i=1:I
        Li{i} = Li{i}+r;
    end

    tk=1;
    Q_old = Q;
    Q_tilde = cell(1,I);
    for it=1:MaxIt
        
        
        
        disp(['at iteration ',num2str(it)])
        
        tic
        for i=1:I
            
            if EXTRA ==1 % Nesterov's accelerate Gradient
                Q_old{i} = Q{i};
                tk = 1;
                for inner_it=1:T
                    if inner_it >1
                        tk_plus_one = (1+sqrt(1+4*tk^2))/2;
                        Q_tilde{i} = Q{i} + (tk-1)/tk_plus_one*(Q{i}-Q_old{i});
                        Q_old{i} = Q{i};
                        Q{i} = Q_tilde{i}-(1/Li{i})*((1/L)*X{i}'*(X{i}*Q_tilde{i})+r*Q_tilde{i}-(1/sqrt(L))*X{i}'*G);
                        
                        if L21==1 % FISTA 
                        Q{i}  = (shrinkL1Lp(Q{i},(1/Li{i})*lambda(i)*ones(1,M),2)); 
                        elseif L11==1
                        Q{i}  = (shrinkL1Lp(Q{i},(1/Li{i})*lambda(i),1)); 
                        end
                        
                        tk = tk_plus_one;
                    else
                        Q{i}=Q{i}-(1/Li{i})*((1/L)*X{i}'*(X{i}*Q{i})+r*Q{i}-(1/sqrt(L))*X{i}'*G);
                        if L21==1
                            Q{i}  = (shrinkL1Lp(Q{i},(1/Li{i})*lambda(i)*ones(1,M(i)),2));
                        elseif L11==1
                            Q{i}  = (shrinkL1Lp(Q{i},(1/Li{i})*lambda(i),1));
                        end
                    end
                    
                    if inner_it>1&&sqrt(sum(sum((Q{i}-Q_old{i}).^2)))<1e-8;
                        break;
                    end
                    
                end
            else
                for inner_it=1:T % Gradient Descent
                    Q{i}=Q{i}-(1/Li{i})*((1/L)*X{i}'*(X{i}*Q{i})+r*Q{i}-(1/sqrt(L))*X{i}'*G);
                    if L21==1 % ISTA
                        Q{i}  = (shrinkL1Lp(Q{i},(1/Li{i})*lambda(i)*ones(1,M(i)),2));
                    elseif L11==1
                        Q{i}  = (shrinkL1Lp(Q{i},(1/Li{i})*lambda(i),1));
                    end
                end
            end
            
        end

        M_temp = zeros(L,K);
        for i=1:I
            XQ{i}= (1/sqrt(L))*X{i}*Q{i};
            M_temp = M_temp + XQ{i};
        end
        
        if L21 ==1||L11==1
            M_temp = 0.9999*M_temp/I  + 0.0001*G;
            
        else
            M_temp = M_temp/I;
        end
        
        % SVD version - global optimality guaranteed
        
        [Ut,St,Vt]=svd(M_temp,0);
        G = Ut(:,1:K)*Vt';
        % subspace_distance = norm(Ut(:,K+1:end)'*G, 2)
        time_perit(it) = toc;
        
        time_acc(it)=sum(time_perit);
        
        obj_temp = 0;
        switch REG_TYPE
            case 'fro'
                for i=1:I
                    obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+ (r/2)*sum(sum(Q{i}.^2)) + obj_temp;
                    Q_norm=(sqrt(sum(Q{i}.^2,2)));
                    q_mean = mean(Q_norm);
                    th = 0.1*q_mean;
                    q_length(i) = length(Q_norm(Q_norm>th));
                end
                obj(it)=sum(obj_temp);
    %             q_length(i)=0;
            case 'none'
                for i=1:I
                    obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2)) + obj_temp;
                    Q_norm=(sqrt(sum(Q{i}.^2,2)));
                    q_mean = mean(Q_norm);
                    th = 0.1*q_mean;
                    q_length(i) = length(Q_norm(Q_norm>th));
                end
                obj(it)=sum(obj_temp);
            case 'L21'
                for i=1:I
                    obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+ (r/2)*sum(sum(Q{i}.^2))+ (lambda(i))*sum(sqrt(sum(Q{i}.^2,2))) + obj_temp;
                    Q_norm=(sqrt(sum(Q{i}.^2,2)));
                    q_mean = mean(Q_norm);
                    th = 0.1*q_mean;
                    q_length(i) = length(Q_norm(Q_norm>th));
                end
                obj(it)=sum(obj_temp);
            case 'L11'
                for i=1:I
                    obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+(r/2)*sum(sum(Q{i}.^2))+ (lambda(i))*sum(sum(abs(Q{i}))) + obj_temp;
                
                    Q_abs = abs(Q{i});
                    q_mean = sum(sum(Q_abs))/(K*M(i));
                    th = 0.1*q_mean;
                    q_length(i) = length(Q_abs(Q_abs>th)); 
                end
                obj(it)=sum(obj_temp);
    %             q_length(i)=0;
        end
        disp(['obj: ', num2str(obj(it))]);
        
        if isempty(Um)~=1
            
            dist(it) = norm(Um'*G,2);
            dist(it)
            
        else
            dist = [];
        end
        if it>1&&abs(obj(it)-obj(it-1))<1e-12
            break;
        end
        
    %     file_name = ['/export/scratch2/xiao/PAMI_MAXVAR/evaluation/I4_M100/',REG_TYPE,'_svd_M100_K',num2str(K),'_iter_',num2str(it)];
    %     save(file_name,'obj_0','obj','time_acc','Q','G','q_length')
        
        for i=1:I
        % disp(['view_',num2str(i), ' obj_', num2str(obj(it))])
        % disp(['the sparsity is ',num2str(q_length(i)/M(i))]) %/(M(i)*K)
        end 
    end

    obj = [obj_0,obj];

    if isempty(dist)
        dist=obj;
    else
        dist = [dist_0,dist];
    end
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
        C2 = max(abs(C1)-lambda,0) .* sign(C1);
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


