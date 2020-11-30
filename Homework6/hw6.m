%% Problem 1
% parameters
n = 1000;
r = 100;
g = 20;
Smean = zeros(1,g);
smean = zeros(1,g);
param = 4/g:4/g:4;
node = 1:n;

% compute Smean and smean
for k = 1:g
    z = param(k);
    p = z/(n-1);
    counter = 0;
    Sdata = zeros(1,r);
    sdata = zeros(1,r);
    
    while counter < r
        G = graph(Erdos_Renyi_Graph(n, p));
        E = dfs(G);
        [bin,size] = conncomp(E);
        [Sdata(1,counter+1), int_max] = max(size);
        
        non_giant_node = node(~(bin == int_max));
        v = randsample(non_giant_node, 1);
        ind_component = bin(v);
        sdata(counter+1) = size(ind_component);
        
        counter = counter + 1;
    end
    
    Smean(k) = mean(Sdata);
    smean(k) = mean(sdata);
end

% compute Sext and sext
Sext = zeros(1,g);

for k = 1:g
    z = param(k);
    syms x
    eqnLeft = x;
    eqnRight = 1 - exp(-z*x);
    Sext(k) = vpasolve(eqnLeft == eqnRight, x, 1);
end

sext = 1./(1 - param + param.*Sext);

% plot

figure(1)
plot(param, Smean./n, '-s','LineWidth',3);
hold on
plot(param, Sext, '--o','LineWidth',3);
legend('empirical','theoretical');
xlabel('z');
ylabel('S(z)')
title('fraction of the giant component')

figure(2)
loglog(param, smean, '-s', 'LineWidth',3);
hold on
loglog(param, sext, '-s', 'LineWidth',3);
legend('empirical','theoretical');
xlabel('z');
ylabel('s(z)')
title('size of a random non-giant component')

%% Problem 2
lmean = zeros(1,4);

for p = 10:13
    n = 2^p;
    q = 4/(n-1);
    r = 100;
    A = Erdos_Renyi_Graph(n, q);
    G = graph(A);
    v = randi([1 n],1,r);
    L = zeros(1,r);

    for k = 1:r
        s = v(k);
        [L(k), ~] = bfs(G,s);
    end
    
    lmean(p-9) = mean(L);
end

% plot
lest = (10:13)*log(2)/log(4);

figure(3)
plot(10:13, lmean,'-s','LineWidth',3);
hold on
plot(10:13, lest,'--o','LineWidth',3);
legend('empirical','theoretical');
xlabel('n');
ylabel('l(n)')
title('Shortest path length')