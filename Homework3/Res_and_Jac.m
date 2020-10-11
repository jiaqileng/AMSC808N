function [r,J] = Res_and_Jac(par,X)
n = length(X);
npar = length(par);
[v,W,u] = param(par);
[fun,dfun,d2fun,~] = ActivationFun();
r = zeros(n,1);
J = zeros(n,npar);
for i = 1 : n
    [ri,dri] = res(X(:,i),v,W,u,fun,dfun,d2fun);
    r(i) = ri;
    J(i,:) = dri';
end
