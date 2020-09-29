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
batch_start = 1;
fun_stats = lossfun(Y,w0,lambda);

if step_size_flag == 1
    m0 = 50;
    index = 0;
    marker = 1;
    step_size = 1;
end

iter = 1;
while iter < iter_max + 1
batch_end = batch_start + batch_size - 1;
batch = batch_start:batch_end;
diff = batch_end - N;
if diff > 0
    batch(end-diff+1:end) = batch(end-diff+1:end) - N;
end

grad = 0;
for j = 1:batch_size
    grad = grad + comp_grad(Y(batch(j),:),w0,lambda);
end

if step_size_flag == 0
    step_size = 1./iter;
elseif step_size_flag == 1
    if iter - marker > 2^index./(index+1)*m0
        marker = iter;
        index = index + 1;
        step_size = step_size./2.^index;
    end
end

grad = grad./batch_size;
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

if batch_end + 1 > N
    batch_start = 1;
else
    batch_start = batch_end + 1;
end

end



end

function f = lossfun(Y,w,lambda)
% Y is a N x 4 row vector
% w is a 4 x 1 column vector

N = size(Y,1);
prod = Y * w; % N x 1 column
f1 = sum(log(1 + exp(-prod)))./N;
f2 = lambda .* w' * w./2;
f = f1 + f2;
end

function g = comp_grad(Y,w,lambda)
% Y is a 1 x 4 row vector
% w is a 4 x 1 column vector

g = - Y'./(1+exp(Y * w)) + lambda .* w;

end