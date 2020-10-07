function [fun_val, gnorm] = stochasticLBFGS(x, Y, lam, m, M,...
                                            batchsize_gradient, batchsize_hessian, maxiter,...
                                            stepsize_flag)
% Input: 
% x = initial guess;
% Y = N x 4 data matrix, dataset;
% lam = Tikhonov regularizer;
% m = memory constant. Usually set as 5;
% M = steps between updating inverse Hessian;
% batchsize_gradient = batch size for sampling gradient, must be <= N;
% batchsize_hessian = batch size for sampling Hessian, must be <= N;
% maxiter = maximal number of iterations;
% snapshot_every = number of iterations between estimating fun. average;
% stepsize_flag = stepsize decreasing strategy;


    N = size(Y,1); % size of dataset
    batchsize_gradient = min(batchsize_gradient, N);
    batchsize_hessian = min(batchsize_hessian, N);
    iter = 1;
    
    if stepsize_flag == 1
        m0 = 50;
        index = 0;
        marker = 1;
        step_size = 1;
    end

    % set (s,y) LBFGS memory
    s = zeros(4,m);
    y = zeros(4,m);
    rho = zeros(1,m);
    gnorm = zeros(1,maxiter);
    fun_val = zeros(1,maxiter);
    
    % first do gradient descend
    g = gfun0(1:N,Y,x,lam);
    a = 0.1;
    xnew = x - a.*g;
    gnew = gfun0(1:N,Y,xnew,lam);
    s(:,1) = xnew - x;
    y(:,1) = gnew - g;
    rho(1) = 1/(s(:,1)'*y(:,1));
    x = xnew;
    g = gnew;
    nor = norm(g);
    gnorm(1) = nor;
    fun_val(1) = lossfun(Y,x,lam);

    % stochastic sg + LBFGS
    while iter < maxiter 
        
        % set step size
        if stepsize_flag == 0
        step_size = 1./(1 + iter./20);
        elseif stepsize_flag == 1
            if iter - marker > 2^index./(index+1)*m0
                marker = iter;
                index = index + 1;
                step_size = step_size./2.^index;
            end
        end
        
        % find update direction 
        if iter < m * M
            upbd = ceil(iter./M);
            I = 1 : upbd;
            p = finddirection(g,s(:,I),y(:,I),rho(I));
        else
            p = finddirection(g,s,y,rho);
        end

        step = step_size .* p;
        xnew = x + step;

        if mod(iter, M) == 0
            s = circshift(s,[0,1]); 
            y = circshift(y,[0,1]);
            rho = circshift(rho,[0,1]);
            hess_seed1 = randi([1 N], 1, batchsize_hessian);
            hess_seed2 = randi([1 N], 1, batchsize_hessian);
            g = gfun0(hess_seed1,Y,x,lam);
            gnew = gfun0(hess_seed2,Y,xnew,lam);
            s(:,1) = step;
            y(:,1) = gnew - g;
            rho(1) = 1/(step'*y(:,1));
        else
            grad_seed = randi([1 N], 1, batchsize_gradient); % generate gradient batch
            gnew = gfun0(grad_seed,Y,xnew,lam); % generate stoc. grad.
        end
        

        
        nor = norm(gnew);
        if nor > .5
            gnorm(iter+1) = norm(g);
            fun_val(iter+1) = lossfun(Y,x,lam);    
        else
            x = xnew;
            g = gnew;
            gnorm(iter+1) = nor;
            fun_val(iter+1) = lossfun(Y,x,lam);
        end
        
        
        iter = iter + 1;
    end

end