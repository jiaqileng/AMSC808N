% We pick k = 5, a = 7.
[M,y] = readdata();
[m,n] = size(M);
[~,S,V] = svd(M,'econ');

%%
ind = readmatrix('pos_selection.txt');
[M,y] = readdata();
M_select = M(:,ind);
[m,n] = size(M_select);
[~,S,V] = svd(M_select,'econ');
k = 10;
score = zeros(1,n);

for j = 1:n
    score(j) = sum(V(j,1:k).^2)./k; % j-th normalized statistical leverage score
end

[B,index] = maxk(score,k)
