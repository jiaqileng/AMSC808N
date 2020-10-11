function [fall,norg] = LevenbergMarquardt(n,N,tol,iter_max)
fsz = 16; % fontsize
%%
Rmax = 1;
Rmin = 1e-14;
rho_good = 0.75;
rho_bad = 0.25;
eta = 0.01;
% iter_max = 10000;
% tol = 5e-3;
%% setup training mesh
n = 10;
a = 1./n;
X = a:a:1;
%% initial guess for parameters
N = 10; % the number of hidden nodes
npar = 3*N;
w = ones(npar,1);
%%
[r,J] = Res_and_Jac(w,X);
f = F(r);
g = J'*r;
nor = norm(g);
R = Rmax/5; % initial trust region radius

fprintf('Initially: f = %d, nor(g) = %d\n',f,nor); 
%% The trust region BFGS method
tic

iter = 0;
flag = 1;
I = eye(length(w));
% quadratic model: m(p) = (1/2)||r||^2 + p'*J'*r + (1/2)*p'*J'*J*p;
norg = zeros(iter_max+1,0);
fall = zeros(iter_max+1,0);
norg(1) = nor;
fall(1) = f;
while nor > tol && iter < iter_max
    % solve the constrained minimization problem using dogleg strategy
    B = J'*J + (1e-12)*I;
    pstar = -B\g;
    fprintf('iter %d: ',iter);
    if norm(pstar) <= R
        p = pstar;
        fprintf('Global min of quad model\n');
    else % solve constrained minimization problem
        lam = 1;
        isub = 0;
        while 1
            B1 = B + lam*I;
            C = chol(B1);
            p = -C\(C'\g);
            np = norm(p);
            dd = abs(np - R);
            if dd < 1e-6
                break
            end
            q = C'\p;
            nq = norm(q);
            lamnew = lam +(np/nq)^2*(np - R)/R;
            if lamnew < 0
                lam = 0.5*lam;
            else
                lam = lamnew;
            end
            isub = isub + 1;
        end
        fprintf('Contraint minimization: %d substeps\n',isub);
    end
    iter = iter + 1;  
    if flag == 0
        break;
    end
    % assess the progress
    wnew = w + p;
    [rnew, Jnew] = Res_and_Jac(wnew,X);
    mnew = 0.5*r'*r + g'*p + 0.5*p'*B*p;
    fnew = F(rnew);
    rho = (f - fnew + 1e-14)/(f - mnew + 1e-14);
    

    % adjust the trust region
    if rho < rho_bad
        R = max([0.25*R,Rmin]);
    else
        if rho > rho_good && abs(norm(p) - R) < tol
            R = min([Rmax,2*R]);
        end
    end
    % accept or reject step
    if rho > eta            
        w = wnew;
        r = rnew;
        J = Jnew;
        f = fnew;
        g = J'*r;
        nor = norm(g);        
%         fprintf('iter # %d: f = %.14f, |df| = %.4e, rho = %.4e, R = %.4e\n',iter,f,nor,rho,R);
    end
    norg(iter+1) = nor;
    fall(iter+1) = f;
end
fprintf('iter # %d: f = %.14f, |df| = %.4e, rho = %.4e, R = %.4e\n',iter,f,nor,rho,R);
cputime = toc;
fprintf('CPUtime = %d, iter = %d\n',cputime,iter);
%% visualize the solution
nt = 201;
t = linspace(0,2,nt);
[fun,~,~,~] = ActivationFun();
[v,W,u] = param(w);
NNfun = zeros(1,nt);
for i = 1 : nt
    NNfun(i) = v'*fun(W.*t(i) + u);
end
sol = 1 + t.*NNfun;
exact_sol = @(y) exp(-y.^2./2)./(1+y+y.^3) + y.^2;
esol = exact_sol(t);
err = sol - esol;
fprintf('max|err| = %d, L2 err = %d\n',max(max(abs(err))),norm(err(:)));

figure;clf
plot(t, err);
xlabel('x');
ylabel('Solution accuracy');
title('Accuracy of the computed solution')
end

%%
function f = F(r)
    f = 0.5*r'*r;
end
