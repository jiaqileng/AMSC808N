%% Part II. 
alpha = 2.2;

% S = the fraction of the giant component
syms x
eqnLeft = x;
eqnRight = polylog(alpha-1,x)./(x.*polylog(alpha-1,1));
u = vpasolve(eqnLeft == eqnRight, x);
S = 1 - polylog(alpha,u)./polylog(alpha,1);
fprintf('u is the smallest non-negative solution to x=G1(x).\n u = %d.\n', double(u));
fprintf('S is the fraction of the giant component.\n S = %d.\n', double(S));
%%
% figure;
% fplot([eqnLeft eqnRight], [0 1],'LineWidth',3)
% hold on
% plot(u, u, 'ko','MarkerSize',10);
% legend('x','G1(x)','soln.')
%% ST = the fraction of infected nodes
T = 0.4;
eqnRightT = polylog(alpha-1,1-T+T.*x)./((1-T+T.*x).*polylog(alpha-1,1));

uT = vpasolve(eqnLeft == eqnRightT, x);
ST = 1 - polylog(alpha,1-T+T.*u)./polylog(alpha,1);
fprintf('u(T) is the smallest non-negative solution to x=G1(x;T).\n u(T) = %d.\n', double(uT));
fprintf('S(T) is the fraction of the infected nodes.\n S(T) = %d.\n', double(ST));
% figure;
% fplot([eqnLeft eqnRightT], [0 1],'LineWidth',3)

%% Part III.
M = 100; % num of random graphs
n = 1e4; % num of nodes of the graph
alpha = 2.2; 

%% average fraction of giant components
fraction_sample = zeros(1,M);

for jj = 1:M
    [A, edges, K, p] = MakePowerLawRandomGraph(n,alpha);
    G = graph(A);
    [~, binsizes] = conncomp(G);
    fraction_sample(jj) = max(binsizes)./n;
end
fraction_average = mean(fraction_sample);
fprintf('The average faction of vertices in the giant component is %d.\n', fraction_average);

%% average fraction of an epidemic
T = 0.4;
epifraction_sample = zeros(1,M);
for jj = 1:M
    [A, edges, K, p] = MakePowerLawRandomGraph(n,alpha);
    [s_trans,t_trans] = transmitting_edges(edges, T);
    G = graph(s_trans, t_trans);
    [~, binsizes] = conncomp(G);
    epifraction_sample(jj) = max(binsizes)./n;
end
epifraction_average = mean(epifraction_sample);
fprintf('The average faction of the epidemic is %d.\n', epifraction_average);

%% critical value of T
T = 0.01:0.01:0.2;
frac_fun = zeros(1,20);

for tt = 1:20
epifraction_sample = zeros(1,M);
for jj = 1:M
    [~, edges, ~, ~] = MakePowerLawRandomGraph(n,alpha);
    [s_trans,t_trans] = transmitting_edges(edges, T(tt));
    G = graph(s_trans, t_trans);
    [~, binsizes] = conncomp(G);
    epifraction_sample(jj) = max(binsizes)./n;
end
frac_fun(tt) = mean(epifraction_sample);
fprintf('When T = %d, The average faction of the epidemic is %d.\n', T(tt), frac_fun(tt));
end
%
figure;
plot(T, frac_fun,'-o','LineWidth',3);
hold on
plot([0.04, 0.04],[0 0.15],'r--');
xlabel('T');
ylabel('fraction of epidemic');
title('Phase transtion');

%% Part IV. 
T = 0.4;
alpha = 2.2;
n = 2e2;

[A, edges, ~, ~] = MakePowerLawRandomGraph(n,alpha);
G = graph(A);

%%
counter = 0;
M = 100;
time_span = 1000;
infection_sample = zeros(M,time_span);

while counter < M
source = randi(n);
s = source;
infection = zeros(1,time_span);
color = cell(n,1);
color(:) = {'white'};
color = string(color);
vertex = 1:n;
%
for k = 1:time_span
    %fprintf('step = %d.\n', k);
    for g = s % s is the node being infected
        color(g) = 'gray';
        nbhd = neighbors(G,g);
        for node = nbhd
            coin = binornd(1,T); % Bernoulli coin
            if coin == 1
                color(node) = 'black'; 
            else
                continue;
            end
        end
    end
    ind_black = (color == "black");
    ind_gray = (color == "gray");
    infection(k) = sum(ind_black) + sum(ind_gray);
    % reset gray node: recover from infection
    color(ind_gray) = "white";
    s = vertex(ind_black);
end

infection_sample(counter+1, :) = infection;
counter = counter + 1
end
% Run M times
figure;
mean_infection = mean(infection_sample, 1);
plot(mean_infection./n);
xlabel('time step')
ylabel('fraction of infected nodes')
title('discrete-time SIR')

%%
T = 0.4;
alpha = 2.2;
max_time = 1000;
M = 100;

duration = zeros(1,20);
dim = 10:10:200;

for jj = 1:20
    n = dim(jj)
    [A, ~, ~, ~] = MakePowerLawRandomGraph(n,alpha);
    G = graph(A);
    duration(jj) = get_duration(G, T, M, max_time);
end

%%
figure;
semilogy(dim, duration,'-o','LineWidth',3)
param = polyfit(dim, log(duration),1);
hold on
semilogy(dim, exp(param(1).*dim + param(2)), 'r--');
xlabel('num of nodes')
ylabel('duration of epidemic')
legend('data','linear fit')
title('How long is the epidemic?','FontSize',10)
%%
function [s, t] = transmitting_edges(edges, T)
[m,~] = size(edges);
counter = 0;
s = [];
t = [];
for k = 1:m
    seed = binornd(1,T);
    if seed == 1
        counter = counter + 1;
        s(counter) = edges(k,1);
        t(counter) = edges(k,2);
    else
        continue;
    end
end
end

function duration = get_duration(G, T, M, max_time)
n = numnodes(G);
counter = 0;
infection_sample = zeros(M,max_time);

while counter < M
source = randi(n);
s = source;
infection = zeros(1,max_time);
color = cell(n,1);
color(:) = {'white'};
color = string(color);
vertex = 1:n;
%
for k = 1:max_time
    %fprintf('step = %d.\n', k);
    for g = s % s is the node being infected
        color(g) = 'gray';
        nbhd = neighbors(G,g);
        for node = nbhd
            coin = binornd(1,T); % Bernoulli coin
            if coin == 1
                color(node) = 'black'; 
            else
                continue;
            end
        end
    end
    ind_black = (color == "black");
    ind_gray = (color == "gray");
    infection(k) = sum(ind_black) + sum(ind_gray);
    % reset gray node: recover from infection
    color(ind_gray) = "white";
    s = vertex(ind_black);
end

infection_sample(counter+1, :) = infection;
counter = counter + 1;
end

mean_infection = mean(infection_sample, 1);
duration = find(mean_infection == 0,1);

if isempty(duration)
    duration =  max_time;
end

end