function [ Q, G ,obj,dist, St, time] = LargeGCCA_distributed_stochastic( X,K, varargin )

    %MaxIt,G,Q,Li,EXTRA,WZW,norm_vec,vec_ind


    % Use 40 gigabit ethernet. Has an effective speed of 4.1 Gbps.
    communication_speed = 4402341478.4    % this is in bps 

    if (nargin-length(varargin)) ~= 2
        error('Wrong number of required parameters');
    end

    %--------------------------------------------------------------
    % Set the defaults for the optional parameters
    %--------------------------------------------------------------

    % Initialize parameters
    [~,I]=size(X);
    for i=1:I
        [L,M(i)]=size(X{i});
    end
    MaxIt = 1000;
    EXTRA = 0;
    Um = [];
    T = 2;
    L11=0; L21=0;r=0;
    Nbits = 3;
    sgd = false;
    compress_g = false;
    compress_avg = false;
    print_log = false;
    distributed = false;
    evaluate = false;
    q_store_interval = 100;
    simulate_comm_latency = false;
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
                    REG_TYPE = varargin{i+1}; %'none','fro'
                case 'UM'
                    Um =  varargin{i+1}; % for measuring error
                case 'NBITS'
                    Nbits = varargin{i+1};
                case 'SGD'
                    sgd = varargin{i+1};
                case 'BATCH_SIZE'
                    batch_size = varargin{i+1};
                case 'RAND_COMPRESS'
                    rand_compress = varargin{i+1};  
                case 'DISTRIBUTED'
                    distributed = varargin{i+1};  
                case 'COMPRESS_G'
                    compress_g = varargin{i+1};
                case 'COMPRESS_AVG'
                    compress_avg = varargin{i+1}
                case 'PRINT_LOG'
                    print_log = varargin{i+1}
                case 'EVALUATE'
                    evaluate = varargin{i+1}
                case 'Q_STORE_INTERVAL'
                    q_store_interval = varargin{i+1}
                case 'SIMULATE_COMM_LATENCY'
                    simulate_comm_latency = varargin{i+1}
                otherwise
                    % Hmmm, something wrong with the parameter string
                    error(['Unrecognized option: ''' varargin{i} '''']);
            end;
        end;
    end

    if evaluate
        Q_store = zeros(I, ceil(MaxIt/q_store_interval)+1, M(1), K);
    end
    for i=1:I
        Q_store(i,1, :,:) = Q{i};
    end
    Nlevels = 2^(Nbits-1) - 1;

    switch REG_TYPE
        case 'none'
            r=0;
        case 'fro'
            r = r;
    end

    obj_temp = 0;
    switch REG_TYPE
        case 'fro'
            for i=1:I
                obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+ (r/2)*sum(sum(Q{i}.^2)) + obj_temp;
            end
        case 'none'
            for i=1:I
                obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+ obj_temp;
            end
    end
    obj_0=sum(obj_temp);

    if isempty(Um)~=1
        dist_0 = norm(Um'*G,2);
    else dist_0=[];
    end

    for i=1:I
        Li{i} = Li{i}+r;
    end

    M_quant = cell(I);
    M_diff = cell(I);
    M_serv = cell(I);

    normalizer = L;
    for i=1:I
        if sgd
            M_serv{i} = (1/sqrt(normalizer))*X{i}*Q{i};
        else
            M_serv{i} = (1/sqrt(L))*X{i}*Q{i};
        end
    end
    G_quant = 0;
    G_client = G;

    M_avg_quant = 0;
    M_temp = zeros(L,K);
    for i=1:i
        M_temp = M_temp + M_serv{i};
    end
    M_temp = M_temp/I;

    M_avg_serv = M_temp;
    M_avg_node = M_avg_serv;
    
    if rand_compress
        compression_scheme = 'qsgd';
    else
        compression_scheme = 'deterministic';
    end
    
    % batch_ind = randperm(L);
    % if sgd
    %     parfor i=1:I
    %         current_batch_id = 1
    %         for inner_it=1:T
    %             current_batch_id
                
    %             if current_batch_id + batch_size -1 > L
    %                 ids = [batch_ind(1:(batch_size-(L-current_batch_id)-1))  batch_ind(current_batch_id:end)];
    %                 % current_batch_id = batch_size - (L-current_batch_id)-1;
    %             else
    %                 ids = batch_ind(current_batch_id:current_batch_id+batch_size-1);
    %                 % current_batch_id = current_batch_id + batch_size
    %             end
    %             current_batch_id = current_batch_id + batch_size;
    %             if current_batch_id > L
    %                 current_batch_id = current_batch_id - L;
    %             end
    %             batch{inner_it}{i} = sparse(X{i}(ids, :)); 
    %             G_batch{inner_it}{i} = G_client(ids, :); 

    %             % batch{inner_it}{i} = sparse(X{i}); 
    %             % G_batch{inner_it}{i} = G_client; 

    %             % ids = batch_ind(current)
    %             % batch{inner_it}{i} = sparse(X{i}( batch_ind(((inner_it-1)*batch_size+1):(inner_it*batch_size)), :)); 
    %             % G_batch{inner_it}{i} = G_client( batch_ind(((inner_it-1)*batch_size+1):(inner_it*batch_size)), :); 

    %             % if inner_it*batch_size > L
    %             %     batch{inner_it}{i} = sparse(X{i}(batch_ind(end-batch_size:end), :));
    %             %     G_batch{inner_it}{i} = G_client(batch_ind(end-batch_size:end), :);
    %             % else
    %             %     (inner_it-1)*batch_size+1
    %             %     (inner_it*batch_size)
    %             %     batch{inner_it}{i} = sparse(X{i}( batch_ind(((inner_it-1)*batch_size+1):(inner_it*batch_size)), :)); 
    %             %     G_batch{inner_it}{i} = G_client( batch_ind(((inner_it-1)*batch_size+1):(inner_it*batch_size)), :); 
    %             % end
    %         end
    %     end
    % end


    tic
    time(1)= 0;
    start_lr = 1;
    final_lr = 1;
    current_lr = 1;
    for it=1:MaxIt
        
        % Node computations
        G_client = G_client + G_quant;
        tic;
        
        current_lr = start_lr - (start_lr-final_lr)*(it/MaxIt);
        if sgd
            for i=1:I
                current_batch_id = 1;
                for inner_it=1:T % Gradient Descent
                    ids = randsample(1:L, batch_size);
                    batch= sparse(X{i}(ids,:)); 
                    G_batch = G_client(ids,:); 

                    % current_lr = 1/it;
                    Q{i}=Q{i}- current_lr*((L/batch_size)*1/Li{i})*((1/normalizer)*batch'*(batch*Q{i})+r*Q{i} -  (1/sqrt(normalizer))*batch'*G_batch);
                end
            end
        else
            for i=1:I  
                for inner_it=1:T % Gradient Descent
                    Q{i}=Q{i} - (1/Li{i})*((1/L)*X{i}'*(X{i}*Q{i})+r*Q{i} - (1/sqrt(L))*X{i}'*G_client);
                end    
            end
        end

        % if sgd
        %     for i=1:I
        %         batch_ind = randperm(L);
        %         batch = X{i};
        %         batch(batch_ind(1:L-batch_size), :) = 0;
        %         for inner_it=1:T % Gradient Descent
        %             Q{i}=Q{i}-(2*1/Li{i})*((1/normalizer)*batch'*(batch*Q{i})+r*Q{i}-(1/sqrt(normalizer))*batch'*G_client);
        %         end
        %     end
        % else
        %     for i=1:I  
        %         for inner_it=1:T % Gradient Descent
        %             Q{i}=Q{i}-(1/Li{i})*((1/L)*X{i}'*(X{i}*Q{i})+r*Q{i}-(1/sqrt(L))*X{i}'*G_client);
        %         end    
        %     end
        % end
        for i=1:I
            % variable to be transmitted
            if sgd
                XQ{i} = (1/sqrt(normalizer))*X{i}*Q{i};
            else
                XQ{i} = (1/sqrt(L))*X{i}*Q{i};
            end
            M_diff{i} = XQ{i} - M_serv{i};
            
            if distributed
                % if rand_compress
                %     % use qsgd
                %     M_quant{i} = qsgd(M_diff{i}, Nbits);
                % else    
                %     % use uniform symmetric quantization 
                %     max_val = max(abs(M_diff{i}),[], 'all');
                %     M_quant{i} = (round((Nlevels/max_val)*M_diff{i})*(max_val/Nlevels));
                % end
                M_quant{i} = compress(M_diff{i}, Nbits, compression_scheme, true);
            else
                M_quant{i} = M_diff{i};
            end
            % % sign quantize
            % M_quant{i} = (norm(M_diff{i},1)/(L*K))*sign(M_diff{i});
            
            % at the server
            M_serv{i} = M_serv{i} + M_quant{i};
        end
        
        time(it+1) = time(it) + toc/I;
        
        node_time = toc/I;

        % Pause twice, for downlink and uplink respectively
        comm_time = 0;
        if simulate_comm_latency
            if distributed
                comm_time = 2*L*K*Nbits / communication_speed;
                % pause(comm_time)
            else
                comm_time = 2*L*K*32 / communication_speed;
                % pause(comm_time)
            end
        end


        tic;
        
        % Server Computation
        M_temp = zeros(L,K);
        
        for i=1:I
            M_temp = M_temp + M_serv{i};
        end
        M_temp = M_temp/I;
        
        if distributed && compress_avg
            M_avg_quant = compress(M_temp-M_avg_serv, Nbits, compression_scheme, true);
        else
            M_avg_quant = M_temp - M_avg_serv;
        end
        
        M_avg_serv = M_avg_serv + M_avg_quant;
        server_time = toc;
        
        tic;
        [Ut,St,Vt]=svd(M_avg_serv,0);
        G = Ut(:,1:K)*Vt';
        svd_time = toc;

        
        tic;
        if distributed && compress_g 
            G_quant = compress(G-G_client, Nbits, compression_scheme, true);
        else
            G_quant = G - G_client;
        end
        compression_time = toc;

        % time_acc(it)=sum(time_perit);
        server_time = server_time + svd_time + compression_time + toc;
        time(it+1) = time(it+1) + toc;


        obj_temp = 0;
        switch REG_TYPE
            case 'fro'
                for i=1:I
                    obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2))+ (r/2)*sum(sum(Q{i}.^2)) + obj_temp;
                end
                obj(it)=sum(obj_temp);
            case 'none'
                for i=1:I
                    obj_temp =(1/2)*sum(sum(((1/sqrt(L))*X{i}*Q{i}-G).^2)) + obj_temp;
                end
                obj(it)=sum(obj_temp);
        end
        
        if print_log
            disp(['at iteration ',num2str(it), ", obj:", num2str(obj(it)), ", comm_time:", num2str(comm_time), ", node_time:", num2str(node_time), ", svd_time:", num2str(svd_time), ", server_time:", num2str(server_time), ", compresss:", num2str(compression_time)]);
        end
        
        if isempty(Um)~=1
            dist(it) = norm(Um'*G,2);
        else
            dist = [];
        end

        if it>1 && abs(obj(it)-obj(it-1))<1e-12
            disp(['Objective value converged. Exiting now.'])
            break;
        end

        if evaluate && rem(it,q_store_interval)==0
            for view=1:I
                Q_store(view, ceil(it/q_store_interval)+1, :, :) = Q{view};
            end
        end
        
    end
    if evaluate
        Q = Q_store;
    end

    obj = [obj_0,obj];
    % time = [0, time];
    if isempty(dist)
        dist=obj;
    else
        dist = [dist_0,dist];
    end
end


function quant = compress(diff, nbits, type, use_max_norm)
    if nargin < 4
        use_max_norm = true;
    end

    if use_max_norm
        ref_val = max(abs(diff),[], 'all');
    end

    [L, K] = size(diff);

    nlevels = 2^(nbits-1) - 1;

    if strcmp(type, 'qsgd')
        quant = qsgd(diff, nbits);
    elseif strcmp(type, 'signsgd')
        quant = (norm(diff,1)/(L*K))*sign(diff);
    elseif strcmp(type, 'deterministic')
        quant = (round((nlevels/ref_val)*diff)*(ref_val/nlevels));
    % elseif strcmp(type, 'qsparse')
    %     quant = 
    end

end

