function [aroc, nn_freq] = eval_europarl(X, Q)
    % @params
    %   X: cell variable containing all views
    %   Q: cell variable containing transformations for all views
    
    % X{1} = [3, 0; 30, 0; 50, 0;];
    % X{2} = [3, 0; 25, 0; 49, 0;];
    % X{3} = [4, 0; 45, 0; 30, 0;];
    
    % for i=1:3
    %     Q{i} = [1 0; 0 0;];
    % end
    
    [J, N] = size(X{1});
    I = length(X);

    % Transformed views
    for i=1:I
        M{i} = X{i}*Q{i};
    end
    
    i = 1;
    a = 2;
    j = 1;

    p = ones(I,I,J);
    q = zeros(I,I,J);
    for i=1:I
        for a=i+1:I
            for j=1:J
                rep_M = repmat(M{i}(j, :), J, 1);
                dist_mat = vecnorm(M{a}-rep_M, 2, 2);
                [~, id] = sort(dist_mat);
                for id_index=1:length(id)
                    if id(id_index) == j
                        p(i,a,j) = id_index;
                        if id_index == 1
                            q(i,a,j) = 1;
                        end
                        break
                    end
                end
            end
        end
    end

    total_sum = 0;
    count = 0;
    for i=1:I
        for a=i+1:I
            total_sum = total_sum + mean(p(i,a,:));
            count = count + 1;
        end
    end
    aroc =( 1 -  (total_sum/count)/J)*100;

    total_sum = 0;
    count = 0;
    for i=1:I
        for a=i+1:I
            total_sum = total_sum + sum(q(i,a,:));
            count = count + 1;
        end
    end
    nn_freq =((total_sum/count)/J)*100;
end