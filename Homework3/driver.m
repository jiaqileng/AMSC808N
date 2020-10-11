nt = 10; % trial mesh is nt-by-nt
N = 20; % the number of neurons
tol = 5e-3; % stop if ||J^\top r|| <= tol
iter_max = 40;  % max number of iterations allowed
[LMf,LMg] = LevenbergMarquardt(nt,N,tol,iter_max);