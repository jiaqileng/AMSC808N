A = readmatrix('Movierankings36.csv');
%%
rows = [1,2,3,4,8,13,16,25,29];

columns = find(isfinite(A(1,:)) & isfinite(A(2,:)) & isfinite(A(3,:)) ...
    & isfinite(A(4,:)) & isfinite(A(8,:)) & isfinite(A(13,:)) ...
    & isfinite(A(16,:)) & isfinite(A(25,:)) & isfinite(A(29,:)));

A = A(rows,columns);
[m,n] = size(A);
A = dlarray(A);
%% PGD method
k = 9;
a = 5e-3; % stepsize
L = 100; % total number of iteration
%W = dlarray(rand(m,k));
%H = dlarray(rand(k,n));
W = dlarray(ones(m,k));
H = dlarray(ones(k,n));

Fnorm = zeros(1,L+1);
R = A - W * H;
Fnorm(1) = sum(R.^2,'all');

for k = 1:L
    Wnew = relu(W + a.*R*H');
    Hnew = relu(H + a.*W'*R);
    W = Wnew;
    H = Hnew;
    R = A - W * H;
    Fnorm(k+1) = sum(R.^2,'all');
end

plot(1:L+1,Fnorm);
%% Lee-Seung scheme
k = 17;
L = 100; % total number of iteration
W = dlarray(rand(m,k));
H = dlarray(rand(k,n));
Fnorm = zeros(1,L+1);
R = A - W * H;
Fnorm(1) = sum(R.^2,'all');

for k = 1:L
    Wnew = (W.*(A*H'))./(W*(H*H'));
    Hnew = (H.*(W'*A))./((W'*W)*H);
    W = Wnew;
    H = Hnew;
    R = A - W * H;
    Fnorm(k+1) = sum(R.^2,'all');
end

plot(1:L+1,Fnorm);
%% Hybrid method
% The first 10 steps in PGD, remaining steps in Lee-Seung.
k = 17;
a = 5e-3; % stepsize
L = 100; % total number of iteration
W = dlarray(rand(m,k));
H = dlarray(rand(k,n));
Fnorm = zeros(1,L+1);
R = A - W * H;
Fnorm(1) = sum(R.^2,'all');

for k = 1:10
    Wnew = relu(W + a.*R*H');
    Hnew = relu(H + a.*W'*R);
    W = Wnew;
    H = Hnew;
    R = A - W * H;
    Fnorm(k+1) = sum(R.^2,'all');
end

for k = 11:L
    Wnew = (W.*(A*H'))./(W*(H*H'));
    Hnew = (H.*(W'*A))./((W'*W)*H);
    W = Wnew;
    H = Hnew;
    R = A - W * H;
    Fnorm(k+1) = sum(R.^2,'all');
end

plot(1:L+1, Fnorm);