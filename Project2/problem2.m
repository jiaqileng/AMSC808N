A_raw = readmatrix('Movierankings36.csv');
[m,n] = size(A_raw);
Omega = isfinite(A_raw);
A = zeros(m,n);
A(Omega > 0) = A_raw(Omega > 0);

%% initialize X,Y
k = 20;
X = rand(m,k);
Y = rand(n,k);
lambda = 20;
L = 1000;
error = zeros(1,L+1);
error(1) = loss(A,Omega,X*Y');
for l = 1:L
    Xnew = updateX(A,Omega,Y,lambda);
    Ynew = updateY(A,Omega,X,lambda);
    X = Xnew;
    Y = Ynew;
    error(l+1) = loss(A,Omega,X*Y');
    
end
plot(1:L+1,error);

%% Nuclear norm trick
M = rand(m,n);
L = 5;
lambda = 0.1;
error = zeros(1,L+1);
error(1) = loss(A,Omega,M);
for l = 1:L
    D = M + (A-M).*Omega;
    [U,S,V] = svd(D);
    Slamb = relu(dlarray(S - lambda.*eye(m,n)));
    M = extractdata(U * Slamb * V');
    error(l+1) = loss(A,Omega,M);
end
plot(1:L+1,error)
%%
function Xnew = updateX(A,Omega,Y,lambda)
[m,~] = size(A);
[~,k] = size(Y);
Xnew = zeros(m,k);
for i = 1:m
    Yi = 0.*Y;
    ai = A(i,:)';
    Yi(Omega(i,:)>0,:) = Y(Omega(i,:)>0,:);
    H = Yi' * Yi + lambda.* eye(k);
    r = Yi' * ai;
    Xnew(i,:) = H\r;
end
end

function Ynew = updateY(A,Omega,X,lambda)
[~,n] = size(A);
[~,k] = size(X);
Ynew = zeros(n,k);
for i = 1:n
    Xi = 0.*X;
    ai = A(:,i);
    Xi(Omega(:,i)>0,:) = X(Omega(:,i)>0,:);
    H = Xi' * Xi + lambda.* eye(k);
    r = Xi' * ai;
    Ynew(i,:) = H\r;
end
end

function val = loss(A,Omega,M)
R = (A - M).*Omega;
val = sum(R.^2,'all');
end
