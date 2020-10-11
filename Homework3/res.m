function [r,dr] = res(x,v,W,u,fun,dfun,d2fun)
% x is a scalar

a = x + (1+3.*x.^2)./(1+x+x.^3);
b = x.^3 + 2.*x + x.^2.*(1+3.*x.^2)./(1+x+x.^3);

[f,fx,df,dfx] = NN(x,v,W,u,fun,dfun,d2fun);

% residual r = Psi_x + a * Psi - b
r = f + x*fx + a*(1+x*f) - b;
% derivative of r w.r.t. parameters
dr = (a*x+1).*df + x.*dfx;
end


