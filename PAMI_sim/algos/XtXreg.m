function v = XtXreg(X,v,lambda);

v = X' * ( X*v ) + lambda*v;

end