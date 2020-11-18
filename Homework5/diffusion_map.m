function Y = diffusion_map(D,epsilon,delta)
%% Diffusion kernel
K = exp(-D./epsilon);

% Convert K into a sotchastic matrix
q = sum(K,2);
Qinv = diag(1./q);
P = Qinv * K;

% Spectral decomposition
[V,D] = eig(P);
d = diag(D);
[~,ind]=sort(abs(d),'descend');

% Embeding into 3D space
R1 = [V(:,ind(2)),V(:,ind(3)),V(:,ind(4))];
lambda1 = d(ind(2));
lambda3 = d(ind(4));
delta = 0.2;
t = ceil(log(delta)./(log(abs(lambda3)) - log(abs(lambda1))));
dt = [d(ind(2)), d(ind(3)), d(ind(4))].^t;
Y = R1 * diag(dt);

% plot
figure();
scatter3(Y(:,1), Y(:,2), Y(:,3),'+');
title('Diffusion Map')
end