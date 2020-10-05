w0 = ones(4,1);
[snapshot_times, fun_average] = sgd(w0,A11,.01,10,10000,100,0);


plot(snapshot_times,fun_average)
%%
function [snapshot_times, fun_average] = sgd(w0,Y,lambda,batch_size,iter_max,run_stats,step_size_flag)
% Input: 
% w0 = initial guess;
% Y = N x 4 data matrix, defined in problem 2;
% lambda = Tikhonov regularizer
% batch_size = batch size per iteration, must be <= N
% iter_max = maximal number of iterations
% run_stats = number of successive samples used for estimating fun. average
% step_size_flag = stepsize decreasing strategy

N = size(Y,1);
fun_stats = lossfun(Y,w0,lambda);

if step_size_flag == 1
    m0 = 50;
    index = 0;
    marker = 1;
    step_size = 1;
end

iter = 1;
while iter <= iter_max % sgd iteration starts
    
batch = randi([1 N], 1, batch_size); % generate batch
grad = stochastic_gradient(batch,Y,w0,lambda); % generate stoc. grad.

% set step size scheme
if step_size_flag == 0
    step_size = 1./iter;
elseif step_size_flag == 1
    if iter - marker > 2^index./(index+1)*m0
        marker = iter;
        index = index + 1;
        step_size = step_size./2.^index;
    end
end


w1 = w0 - step_size .* grad;
fun_stats = fun_stats + lossfun(Y,w1,lambda);
    
if mod(iter,run_stats) == 0
    num_ave = iter./run_stats;
    snapshot_times(num_ave) = iter;
    fun_average(num_ave) = fun_stats./run_stats;
    fun_stats = 0;
end

w0 = w1;
iter = iter + 1;

end
end

