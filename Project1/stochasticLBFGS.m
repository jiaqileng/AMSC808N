x = ones(4,1); % initial vector
Y = A11; % intialize data
N = size(Y,1); % size of dataset
m = 5; % memory constant
batchsize_gradient = 128; 
batchsize_hessian = min(256,N);
M = 5; % steps between updating inverse Hessian
maxiter = 1e3;
iter = 1;
tol = 1e-6;
lambda = .01; % Tikhonov regularizer

s = zeros(4,m);
y = zeros(4,m);
rho = zeros(1,m);
gnorm = zeros(1,maxiter);
fun_eval = zeros(1,maxiter);

% first do gradient descend
g = stochastic_gradient(1:N,Y,x,lambda);
a = 0.05;
xnew = x - a.*g;
gnew = stochastic_gradient(1:N,Y,xnew,lambda);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
x = xnew;
g = gnew;
nor = norm(g);
gnorm(1) = nor;
fun_eval(1) = lossfun(Y,x,lambda);

% stochastic sg + LBFGS
while iter < maxiter % sg iteration to yield approx. grad.
    stepsize = 1./(iter);
    % find update direction 
    if iter < m * M
        upbd = ceil(iter./M);
        I = 1 : upbd;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
        
    step = stepsize .* p;
    xnew = x + step;
    
    if mod(iter, M) == 0
        s = circshift(s,[0,1]); 
        y = circshift(y,[0,1]);
        rho = circshift(rho,[0,1]);
        hess_seed1 = randi([1 N], 1, batchsize_hessian);
        hess_seed2 = randi([1 N], 1, batchsize_hessian);
        g = stochastic_gradient(hess_seed1,Y,x,lambda);
        gnew = stochastic_gradient(hess_seed2,Y,xnew,lambda);
        s(:,1) = step;
        y(:,1) = gnew - g;
        rho(1) = 1/(s(:,1)'*y(:,1));
    else
        grad_seed = randi([1 N], 1, batchsize_gradient); % generate gradient batch
        gnew = stochastic_gradient(grad_seed,Y,xnew,lambda); % generate stoc. grad.
    end
    
    nor = norm(gnew);
    if nor > .5
        gnorm(iter+1) = norm(g);
        fun_eval(iter+1) = lossfun(Y,x,lambda);
    elseif iter > 300 && nor > .1
        gnorm(iter+1) = norm(g);
        fun_eval(iter+1) = lossfun(Y,x,lambda);       
    else
        x = xnew;
        g = gnew;
        gnorm(iter+1) = nor;
        fun_eval(iter+1) = lossfun(Y,x,lambda);
    end
    
    iter = iter + 1;
end
    
figure(1)
plot(1:maxiter, fun_eval)

figure(2)
plot(1:maxiter, gnorm)