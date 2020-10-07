%% Loading data

A2016 = readmatrix('A2016.csv');
% Format for A2012 and A2016:
% FIPS, County, #DEM, #GOP, then <str> up to Unemployment Rate
str = ["Median Income", "Migration Rate", "Birth Rate",...
"Death Rate", "Bachelor Rate", "Unemployment Rate","log(#Votes)"];

% remove column county that is read by matlab as NaN
A2016(:,2) = [];

% Remove rows with missing data
A = A2016;
% remove all rows with missing data
ind = find(~isfinite(A(:,2)) |  ~isfinite(A(:,3)) | ~isfinite(A(:,4)) ...
    | ~isfinite(A(:,5)) | ~isfinite(A(:,6)) | ~isfinite(A(:,7)) ...
    | ~isfinite(A(:,8)) | ~isfinite(A(:,9)));
A(ind,:) = [];

% select CA, OR, WA, NJ, NY counties
% ind = find((A(:,1)>=6000 & A(:,1)<=6999)); %...  %CA
%  | (A(:,1)>=53000 & A(:,1)<=53999)); %...        %WA
%  | (A(:,1)>=34000 & A(:,1)<=34999) ...        %NJ  
%  | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
%  | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
% A = A(ind,:);

[n,dim] = size(A);

% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

% select max subset of data with equal numbers of dem and gop counties
ngop = length(igop);
ndem = length(idem);
if ngop > ndem
    rgop = randperm(ngop,ndem);
    Adem = A(idem,:);
    Agop = A(igop(rgop),:);
    A = [Adem;Agop];
else
    rdem = randperm(ndem,ngop);
    Agop = A(igop,:);
    Adem = A(idem(rdem),:);
    A = [Adem;Agop];
end  
[n,dim] = size(A);
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

% set up matrix data and rescale it to [0,1]
X = [A(:,4:9),log(num)];
X(:,1) = X(:,1)/1e4;
% select three data types that distinguish dem and gop counties the most
i1 = 1; % Median Income
i2 = 7; % log(# votes)
i3 = 5; % Bachelor Rate
XX = X(:,[i1,i2,i3]); % data matrix
% rescale all data to [0,1]
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
X1 = (XX(:,1)-xmin)/(xmax-xmin);
X2 = (XX(:,2)-ymin)/(ymax-ymin);
X3 = (XX(:,3)-zmin)/(zmax-zmin);
XX = [X1,X2,X3];

num_data = length(label);
Y = diag(label) * [XX ones(num_data,1)];

%% Problem 1: 

% constraint with soft margins
A12 = eye(num_data);
A21 = zeros(num_data,4);
A = [Y A12;A21 A12];
b = [ones(num_data,1);zeros(num_data,1)];

% find initial guess
[w,l,lcomp] = FindInitGuess(ones(4,1), Y, ones(num_data,1));
x0 = [w;lcomp];

% run active set method
W = [];
C = 1e3;
gfun = @(x) gfun_soft(x,C);
Hfun = @(x) Hfun_soft(x,C);
[xiter,lm] = ASM(x0,gfun,Hfun,A,b,W);
w = xiter(1:4,end);

% plot
plot_plane(XX, w, idem, igop, 'Separating plane for CA data')

%% Problem 2: stochastic SG

w0 = [-1;-1;1;1];
lam = .01;
batch_size = 128;
maxiter = 2e3;
time = zeros(1,100);
fun_sum = zeros(1,maxiter);

for k = 1:1000
    tic
    [fun_val, gnorm] = sgd(w0,Y,lam,batch_size,maxiter,0);
    time(k) = toc;
    fun_sum = fun_sum + fun_val;
    fprintf('k = %d\n',k);
end
fprintf('running time = %d\n', mean(time));
%
fun_average_sgd = fun_sum./1000;
%%
figure;
plot(1:maxiter,fun_average_sgd,'b-','Linewidth',3);
title('average function value v.s iterations')

% add more plots based on Zezheng's writing

%% Problem 3: Subsampled inexact Newton
w0 = [-1;-1;1;1];
fun = @(I,Y,w) fun0(I,Y,w,lam);
gfun = @(I,Y,w) gfun0(I,Y,w,lam);
Hvec = @(I,Y,w,v) Hvec0(I,Y,w,v,lam);
batch_size = 256;
maxiter = 2e3;
fun_sum = zeros(1,maxiter);
time = zeros(1,1000);

for k = 1:1000
    tic
    [w,f,normgrad] = SINewton(fun,gfun,Hvec,batch_size,Y,w0,maxiter);
    time(k) = toc;
    fun_sum = fun_sum + f';
    fprintf('k = %d\n',k);
end
fprintf('running time = %d\n', mean(time));
fun_average_SINewton = fun_sum./1000;
%%
figure;
plot(1:maxiter, fun_average_SINewton);
%%
figure;
plot_plane(XX, w, idem, igop, 'subsampled inexact Newton')
% add more plots based on Chenyang's writing

%% Problem 4: stochastic L-BFGS

x = [-1;-1;1;1]; % initial vector
lam = .01; % Tikhonov regularizer
m = 5; % memory constant

% frequency of updating pairs. 
% We test M = 10, 20, 50.
M = 10;  

% batchs ize for gradient.
% We test 32, 64, 128
batchsize_gradient = 128; 

% batch size for (s,y) pairs.
% We test 64, 128, 256
batchsize_hessian = 256;


maxiter = 2e3;

% run 1000 times, do average
time = zeros(1,1000);
fun_sum = zeros(1,maxiter);
k = 1;
while k < 1001
    tic;
    [fun_val, gnorm] = stochasticLBFGS(x, Y, lam, m, M,batchsize_gradient, batchsize_hessian, maxiter,0);
    time(k) = toc;
    fprintf('k = %d\n', k);
    
    if sum(fun_val) < 1e10
        fun_sum = fun_sum + fun_val;
        k = k + 1;
    else
        continue;
    end
end

fprintf('average running time = %d\n', sum(time)/1000);
%%
fun_average_stochasticLBFGS = fun_sum./1000;
%%
figure;
plot(1:maxiter, fun_average_freq_10,'b-','Linewidth',3);
hold on
plot(1:maxiter, fun_average_freq_20,'r-','Linewidth',3);
plot(1:maxiter, fun_average_freq_50,'g-','Linewidth',3);
legend('M=10','M=20','M=50');
xlabel('no. of iterations');
ylabel('averaged function value');
title('stochastic L-BFGS with updating pairs every M: 10, 20, 50');

%%
figure;
plot(1:maxiter, fun_average_sgd,'b-','Linewidth',3);
hold on
plot(1:maxiter, fun_average_SINewton,'r-','Linewidth',3);
plot(1:maxiter, fun_average_stochasticLBFGS,'g-','Linewidth',1);
legend('SG','SINewton','stochastic L-BFGS');
xlabel('no. of iterations');
ylabel('averaged function value');
title('Performance of three optimization methods');

%%
function g = gfun_soft(x,C)
w = x(1:4);
lam = x(5:end);
l = length(lam);
g = [w;C.*ones(l,1)];
end

function Hv = Hfun_soft(x,C)
lam = x(5:end);

l = length(lam);
Hv = zeros(4+l);
for j = 1:3
    Hv(j,j) = 1;
end
end

function f = fun0(I,Y,w,lam)
f = sum(log(1 + exp(-Y(I,:)*w)))/length(I) + 0.5*lam*w'*w;
end

function Hv = Hvec0(I,Y,w,v,lam)
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
Hv = sum(Y(I,:).*((aux.*(Y(I,:)*v)./((1+aux).^2)).*ones(1,d1)),1)' + lam*v;
end