% We pick k = 5, a = 7.
[M,y] = readdata();
% lableing
florida = find(y == -1);
indiana = find(y == 1);

[m,n] = size(M);
[~,S,V] = svd(M,'econ');
k = 5;
score = zeros(1,n);
for j = 1:n
    score(j) = sum(V(j,1:k).^2)./k; % j-th normalized statistical leverage score
end
fprintf('Indices with 5 maximal leverage scores:\n');
[~,index] = maxk(score,k)
%% pre-selection 
ind = readmatrix('pos_selection.txt'); % Lexical
% ind = stat_selection(M,y); % Statistical
%%
M_select = M(:,ind);
[~,n_select] = size(M_select);
[~,S,V_select] = svd(M_select,'econ');
k = 5;
score_select = zeros(1,n_select);

for j = 1:n_select
    score_select(j) = sum(V_select(j,1:k).^2)./k; % j-th normalized statistical leverage score
end

fprintf('Indices with 5 maximal leverage scores (after pre-selection on POS):\n');
[~,index_select] = maxk(score_select,k);
index_select_original = ind(index_select)
%%
PCA1 = V_select(:,1:2);
projection1 = M_select * PCA1;
figure;
scatter(projection1(florida,1),projection1(florida,2),'r','filled');
hold on
scatter(projection1(indiana,1),projection1(indiana,2),'b','filled');
xlabel('v1');
ylabel('v2');
legend('Florida','Indiana');
title('PCA classification');

%% 
Msmall = M_select(:,index_select);
[~,~,Vsmall] = svd(Msmall,'econ');
PCA2 = Vsmall(:,1:2);
projection2 = Msmall * PCA2;
figure;
scatter(projection2(florida,1),projection2(florida,2),'r','filled');
hold on
scatter(projection2(indiana,1),projection2(indiana,2),'b','filled');
xlabel('v1');
ylabel('v2');
legend('Florida','Indiana');
title('PCA classification');

%%
function ind_stat = stat_selection(M,y)
florida = find(y == -1);
indiana = find(y == 1);
PF = length(florida);
PI = length(indiana);

score_stat = abs(sum(M(florida,:))./PF - sum(M(indiana,:))./PI);
[~,ind_stat] = maxk(score_stat,10000);
ind_stat = sort(ind_stat);
end