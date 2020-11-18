%% Load data
% Scurve data
% MakeScurveData();
% dat = load('ScurveData.mat');

% Face data
MakeEmojiData()
dat = load('FaceData.mat');
X = dat.data3;
[n,~] = size(X);

% % Add Gaussian noise
% noisestd = 0.1;
% X = X + noisestd*randn(size(X)); % perturb by Gaussian noise
% plot3(X(:,1),X(:,2),X(:,3),'.','Markersize',20);
% legend('original','noisy')
%% Dimension reduction

% PCA
m = 2;
Y1 = pca_reduction(X,m);
figure;
scatter(Y1(:,1),Y1(:,2),12,'+');
title('PCA')

%% Isomap
k = 20;
Y2 = isomap(X,k);
figure();
scatter(Y2(:,1),Y2(:,2),12,'+');
title('Isomap')

%% LLE
m = 2;
K = 40;
Y3 = lle(X',K,m);
Y3 = Y3';
figure();
scatter(Y3(:,1),Y3(:,2),12,'+');
title('LLE')

%% t-SNE
Y4 = tsne(X);
figure();
scatter(Y4(:,1),Y4(:,2),12,'+');
title('t-SNE')

%% Diffusion map
% Compute squared-distance matrix
D = zeros(n);
for ii = 1:n
    for jj = 1:n
        D(ii,jj) = norm(X(ii,:) - X(jj,:),2)^2;
    end
end

%% Compute epsilon
drowmin = zeros(1,n);
for k = 1 : n
    drowmin(k) = min(D(k,setdiff(1:n,k)));
end
ep = 2*mean(drowmin)

%% Run diffusion map
epsilon = ep;
delta = 0.2;
%%
Y = diffusion_map(D,200,delta);