function grad = stochastic_gradient(batch,Y,w0,lambda)
batch_size = length(batch);
grad = 0;
for j = 1:batch_size
    y = Y(batch(j),:);
    g = - y'./(1+exp(y * w0)) + lambda .* w0;
    grad = grad + g;
end

grad = grad./batch_size;
end