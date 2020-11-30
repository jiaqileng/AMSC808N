function E = dfs(G)
%% Initialization
N = numnodes(G);
color = cell(N,1);
%parent = cell(N,1);
color(:) = {'white'};
%parent(:) = {'nil'};
color = string(color);
%parent = string(parent);
global T
T = table(color);
T.parent = ones(N,1)*Inf;
T.d = zeros(N,1);
T.f = zeros(N,1);


%% run loop
time = 0;
for u = 1:N
    if T.color(u) == "white"
        time = dfs_visit(G,u,time);
    end
end

ind = ~isinf(T.parent);
nodes = (1:N)';
E = graph(T.parent(ind), nodes(ind),'omitselfloop');
end

function time = dfs_visit(G,u,time)
global T
time = time + 1;
T.d(u) = time;
T.color(u) = "gray";

adjacent_vert = neighbors(G,u);
g = length(adjacent_vert);
for k = 1:g
    v = adjacent_vert(k);
    if T.color(v) == "white"
        T.parent(v) = u;
        time = dfs_visit(G,v,time);
    end
end

T.color(u) = "black";
time = time + 1;
T.f(u) = time;
end