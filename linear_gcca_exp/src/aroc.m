function score = aroc(X, Q)
    % @params
    %   X: cell variable containing all views
    %   Q: cell variable containing transformations for all views
    [J, N] = size(X{1});
    I = length(X);

    % Transformed views
    for i=1:I
        M{i} = X{i}*Q{i}
    end
    
    i = 1;
    a = 2;
    j = 1;

    rep_M = repmat(M{i}(j), J, 1);
    dist_mat = vecnorm(M{i}-rep_M, 2, 2);
    [~, id] = sort(dist_mat);
    id =
    
    % for i=1:I 
    %     for a=1:I 
    %         if ~i==a:
    %             for 
                
end