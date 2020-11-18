function Y = pca_reduction(X, m)
%%
[n, D] = size(X);
m = sum(X,1)./n;
A = X - ones(n,1) * m;
[~,~,V] = svd(A);
m = 2;
Y = X * V(:,1:m);
end