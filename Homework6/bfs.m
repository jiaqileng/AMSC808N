function [l, component] = bfs(G,s)
% Inputs: G is a graph, s is a source vertex
% Outputs: l is the shortest path length in the connected component
% containing s, 'component' is a list of nodes in this component.
%% Initialization
N = numnodes(G);
color = cell(N,1);
parent = cell(N,1);
color(:) = {'white'};
parent(:) = {'nil'};
color = string(color);
parent = string(parent);
T = table(color, parent);
T.d = ones(N,1)*Inf;

%% Start queue
T.color(s) = "gray";
T.d(s) = 0;
Q = [];
Q(1) = s;

while ~isempty(Q)
    u = Q(1);
    adjacent_vert = neighbors(G,u);
    g = length(adjacent_vert);
    for k = 1:g
        v = adjacent_vert(k);
        if T.color(v) == "white"
            T.color(v) = "gray";
            T.d(v) = T.d(u) + 1;
            T.parent(v) = u;
            Q(end+1) = v;
        end
    end
    T.color(u) = "black";
    Q(1) = [];
end

%% return data
ind = ~isinf(T.d);
l = max(T.d(ind));
nodes = 1:N;
component = nodes(ind);
end
