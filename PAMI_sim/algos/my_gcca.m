function Q = my_gcca( X, r,lambda )

% r: rank
% lambda: regularization

[n,~] = size(X{1});

opts.issym  = 1;
opts.isreal = 1;

[Q,~] = eigs( @(u)sumX(X,u,n,lambda), n, r, 'LM', opts);

end

function u = sumX(X,u,n,lambda)

uu = zeros(n,1);
for i = 1:length(X)
    vv = X{i}'*u;
    vv = pcg_mat( @(v)XtXreg(X{i},v,lambda), vv );
    uu = uu + X{i}*vv;
end

u = uu;
end

function v = XX( X, v )
v = X' * ( X*v );
end