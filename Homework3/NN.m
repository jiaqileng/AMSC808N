function [f,fx,df,dfx] = NN(x,v,W,u,fun,dfun,d2fun)
%% derivatives of the network 
z = W*x + u;
s0 = fun(z); % sigma(z)
f = v'*s0; 
s1 = dfun(z); % sigma'(z)
s2 = d2fun(z); % sigma''(z)
fx = v'*(W.*s1); % Psi_x

%% derivatives with respect to parameters
nv = length(v); 
nw = length(W);
nu = length(u); % nu2 must be 1
dim = nv + nw + nu;
df = zeros(dim,1);
dfx = zeros(dim,1);

% df
df(1:nv) = s0;
df(nv+1 : nv+nw) = x.*(v.*s1);
df(nv+nw+1 : end) = v.*s1;
% dfx
dfx(1:nv) = W.*s1;
dfx(nv+1 : nv+nw) = x.*(v.*W.*s2) + (v.*s1);
dfx(nv+nw+1 : end) = v.*W.*s2;

end