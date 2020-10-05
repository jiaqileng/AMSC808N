function f = lossfun(Y,w,lambda)
% Y is a N x 4 row vector
% w is a 4 x 1 column vector

N = size(Y,1);
prod = Y * w; % N x 1 column
f1 = sum(log(1 + exp(-prod)))./N;
f2 = lambda .* w' * w./2;
f = f1 + f2;
end