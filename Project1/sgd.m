function [fun_val, gnorm] = sgd(x,Y,lam,...
                                             batch_size,maxiter,...
                                             stepsize_flag)
% Input: 
% x = initial guess;
% Y = N x 4 data matrix, defined in problem 2;
% lam = Tikhonov regularizer
% batch_size = batch size per iteration, must be <= N
% maxiter = maximal number of iterations
% snapshot_every = number of iterations between estimating fun. average
% stepsize_flag = stepsize decreasing strategy

    N = size(Y,1);
    fun_val = zeros(1,maxiter);
    fun_val(1) = lossfun(Y,x,lam);
    gnorm = zeros(1,maxiter);
    iter = 1;
    
    if stepsize_flag == 1
        m0 = 50;
        index = 0;
        marker = 1;
        step_size = 1;
    end

    % sgd iteration starts
    while iter < maxiter 

        batch = randi([1 N], 1, batch_size); % generate batch
        g = gfun0(batch,Y,x,lam); % generate stoc. grad.

            % set step size scheme
        if stepsize_flag == 0
        step_size = 1./(1 + iter./20);
        elseif stepsize_flag == 1
            if iter - marker > 2^index./(index+1)*m0
            marker = iter;
            index = index + 1;
            step_size = step_size./2.^index;
            end
        end


        xnew = x - step_size .* g;
        fun_val(iter+1) = lossfun(Y,xnew,lam);
       
    
    gnorm(iter+1) = norm(g);
    x = xnew;
    iter = iter + 1;
    end
end