%% Part I. Plot of Regions
relu = @(x) max(x,0);
g = @(x) 1 - cos(x);

dx = pi/10;
xx = 0:dx:5*dx;
gx = g(xx);

% Compute global minima
l = 4;
xxx = xx(6+1-l:6);
gxx = gx(6+1-l:6);
A = sum(xxx.^2)/sum(xxx);
B = sum(xxx.*gxx)/sum(xxx);
C = sum(xxx)/l;
D = sum(gxx)/l;
amin = (B-D)/(A-C);
bmin = C*amin - D;
fun_min = fun([amin,bmin]);
fprintf('The global minimizer is (%d, %d), with function value %d\n',amin, bmin ,fun_min);

% Region plot
figure;
a = -5:0.1:5;
a1 = -3:0.1:0;
a2 = 0:0.1:2;
a3 = 3.2:0.1:5;
plot(a,0.*a, 'r--','LineWidth',1);
hold on
for k = 1:5
   plot(a,((k/10).*pi).*a, 'r--','LineWidth',1);
end
plot(a3, xx(6).*a3 - gx(6), 'b-','LineWidth',3);
plot(amin,bmin,'o','LineWidth',3)
text(-3,3,'$\mathcal{F}$: the flat region','Interpreter','latex','FontSize',20)
text(3,-3,'$\Omega$','Interpreter','latex','FontSize',20)
xlabel('a')
ylabel('b')
%% Part II.
% plot of the descent step from (1,0)
figure;
g = grad_fun([1,0]);
a = 0:0.1:2;
plot(a, 0.5.*pi.*a, 'r--','LineWidth',3);
hold on
plot(a, 0.*a, 'r--','LineWidth',3);
plot(a, (g(2)/g(1)).*(a-1),'b-','LineWidth',3);
plot(1,0,'ko','LineWidth',5);
text(0.4,2,'$\mathcal{F}$: the flat region','Interpreter','latex','FontSize',20)
annotation('textarrow',[0.5 0.4],[0.5 0.4],'String','descent direction')
xlabel('a')
ylabel('b')

% compute the minimal stepsize to hit the flat region
alpha_min = pi/(pi*g(1) - 2*g(2));
fprintf('The minimal stepsize so that x1 hits the flat region is %d.\n',alpha_min);

%% GD constant stepsize
% M = 10;
M = 1e4;
loss_gd = zeros(1,M+1);
gd_iterate = zeros(M+1,2);
x0 = [1, 0];
% alpha = alpha_min;
alpha = 0.99*alpha_min;
loss_gd(1) = fun(x0);

for ii = 1:M
    g = grad_fun(x0);
    xnew = x0 - alpha .* g;
    gd_iterate(ii+1,:) = xnew;
    loss_gd(ii+1) = fun(xnew);
    x0 = xnew;
end

figure;
%plot(loss_gd,'-x','LineWidth',3)
semilogy(loss_gd)
ylabel('Loss of GD')

title('stepsize = $0.99\alpha^*$','Interpreter','latex','FontSize',20)
%% compute Hessian & eigenvalues
H = hessian_fun([amin, bmin]);
eigval = eig(H);
alpha_max = 2/eigval(2);
fprintf('Maximal convergent stepsize is %d.\n', alpha_max);

figure; % plot of convergence
M = 5e2;
loss_gd = zeros(1,M+1);
gd_iterate = zeros(M+1,2);
x0 = [1, 0];
alpha = 1.3;

loss_gd(1) = fun(x0);
gd_iterate(1,:) = x0;

for ii = 1:M
    g = grad_fun(x0);
    xnew = x0 - alpha .* g;
    gd_iterate(ii+1,:) = xnew;
    loss_gd(ii+1) = fun(xnew);
    x0 = xnew;
end
semilogy(loss_gd,'-x','LineWidth',3)
hold on
semilogy(0*loss_gd+fun_min, 'r--','LineWidth',2)
ylabel('Loss of GD')
legend('Loss curve','global minimum')
title('stepsize = 1.3','Interpreter','latex','FontSize',20)

% iteration path
figure;
plot(gd_iterate(:,1), gd_iterate(:,2),'-o');
hold on
plot(amin, bmin, 'kx','LineWidth',5);
xlabel('a')
ylabel('b')
legend('iteration','global minimizer')

%% Part III. SGD
figure; % plot of SGD
M = 3e3;
loss_sgd = zeros(1,M+1);
sgd_iterate = zeros(M+1,2);
x0 = [1, 0];

loss_sgd(1) = fun(x0);
sgd_iterate(1,:) = x0;

for ii = 1:M
    seed = randi(6,1);
    if ii < 2e3
        alpha = 0.5+0.5/ii;
    else
        alpha = 1/ii;
    end
    g = grad_fun_single(x0,seed);
    xnew = x0 - alpha .* g;
    sgd_iterate(ii+1,:) = xnew;
    loss_sgd(ii+1) = fun(xnew);
    x0 = xnew;
end

semilogy(0*loss_gd+fun_min, 'r--','LineWidth',2)
hold on
semilogy(loss_sgd,'b-','LineWidth',1)

ylabel('Loss of SGD')
legend('global minimum','Loss curve')
title('SGD with $\alpha_k = 1/k$','Interpreter','latex','FontSize',20)

% iteration path
figure;
plot(sgd_iterate(:,1), sgd_iterate(:,2),'-o');
hold on
plot(amin, bmin, 'kx','LineWidth',5);
xlabel('a')
ylabel('b')
legend('iteration','global minimizer')

%%
function val = fun(x0)
a = x0(1);
b = x0(2);
relu = @(x) max(x,0);
g = @(x) 1 - cos(x);
dx = pi/10;
xx = 0:dx:5*dx;
gx = g(xx);
val = 0;
for j = 1:6
    val = val + (relu(a*xx(j) - b) - gx(j))^2;
end

val =  val/12;
end

function g = grad_fun_single(x0 ,j)
a = x0(1);
b = x0(2);
g = @(x) 1 - cos(x);

dx = pi/10;
xx = 0:dx:5*dx;
gx = g(xx);

if b < a*xx(j)
    L = a*xx(j) - b - gx(j);
else
    L = 0;
end

ga = xx(j)*L/6;
gb = -L/6;

g = [ga, gb];
end

function g = grad_fun(x0)
a = x0(1);
b = x0(2);
g = [0,0];

for j = 1:6
    g = g + grad_fun_single([a,b],j);
end

end

function H = hessian_fun(x0)
a = x0(1);
b = x0(2);
g = @(x) 1 - cos(x);

dx = pi/10;
xx = 0:dx:5*dx;
gx = g(xx);

H = zeros(2);
for j = 1:6
    if b < a*xx(j)
        H = H + [xx(j).^2, -xx(j); -xx(j), 1];
    else
        continue;
    end
end
H = H./6;
end