[M,y] = readdata();

[~,S,V1] = svd(M,'econ');
[~,~,V2] = svd(M','econ');
L1 = 10;
L2 = 8;
sample_size = 100;

%%
denomenator = zeros(1,L1-1);
numerator = zeros(L1-1,L2);
sigma = diag(S);

for k = 2:L1
    denomenator(k-1) = sqrt(sum(sigma(k+1:end).^2));
    for a = 1:L2
        c = a*k;
        r = a*k;
        counter = 1;
        sample = zeros(1,sample_size);
        parfor counter = 1:sample_size
            tic
            C = columnselect(M,V1,k,c);
            R = columnselect(M',V2,k,r)';
            U = pinv(C) * M * pinv(R);
            sample(counter) = kyfan2(M - C*U*R);
        end
        numerator(k-1,a) = mean(sample);
        fprintf('k = %d,a = %d, completed!\n',k,a);
    end
end

% save p3_computed.mat numerator denomenator
%%
figure;
for a = 1:L2
    text = sprintf('a = %d',a);
    plot(2:L1, numerator(:,a)'./denomenator,'-s',...
        'DisplayName',text,'LineWidth',2);
    hold on
end
plot(2:L1,ones(1,L1-1),'--','DisplayName','ref');
xlabel('k');
ylabel('ratio');
legend;
title('$\|M-CUR\|_F/\|M-M_k\|_F$','Interpreter','latex')

figure;
for a = 1:L2
    text = sprintf('a = %d',a);
    plot(2:L1, numerator(:,a)','-s',...
        'DisplayName',text,'LineWidth',2);
    hold on
end
xlabel('k');
ylabel('F-norm');
legend;
title('$\|M-CUR\|_F$','Interpreter','latex')
%%
function C = columnselect(A,V,k,c)
[m,n] = size(A);
C = zeros(m,n);
for j = 1:n
    score = sum(V(j,1:k).^2)./k; % j-th normalized statistical leverage score
    p = min(1,c*score);
    sample = binornd(1,p);
    if sample == 1
        C(:,j) = A(:,j);
    end
end
end

function val = kyfan2(A)
% Frobenius norm of A
val = sqrt(sum(A.^2,'all'));
end


function [M,y] = readdata()
%% read data
fid=fopen('vectors.txt','r');
A = fscanf(fid,'%i\n');
ind = find(A>100000); % find entries of A with document IDs
n = length(ind); % the number of documents
la = length(A);
I = 1:la;
II = setdiff(I,ind);
d = max(A(II)); % the number of words in the dictionary
M = zeros(n,d);
y = zeros(n,1); % y = -1 => category 1; y = 1 => category 2 
% define M and y
for j = 1 : n
    i = ind(j); 
    y(j) = A(i+1); 
    if j<n 
        iend = ind(j+1)-1; 
    else
        iend = length(A);
    end
    M(j,A(i+2:2:iend-1)) = 1; A(i+3:2:iend);
end
i1 = find(y==-1);
i2 = find(y==1);
ii = find(M>0);
n1 = length(i1);
n2 = length(i2);

fprintf('Class 1: %d items\nClass 2: %d items\n',n1,n2);
fprintf('M is %d-by-%d, fraction of nonzero entries: %d\n',n,d,length(ii)/(n*d));
end