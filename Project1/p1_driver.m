function problem1_softmargin()
close all
%% Data extraction and Rescaling

% read data
A2012 = readmatrix('A2012.csv');
A2016 = readmatrix('A2016.csv');
% Format for A2012 and A2016:
% FIPS, County, #DEM, #GOP, then <str> up to Unemployment Rate
str = ["Median Income", "Migration Rate", "Birth Rate",...
"Death Rate", "Bachelor Rate", "Unemployment Rate","log(#Votes)"];

% remove column county that is read by matlab as NaN
A2012(:,2) = [];
A2016(:,2) = [];

% Remove rows with missing data
A = A2016;
% remove all rows with missing data
ind = find(~isfinite(A(:,2)) |  ~isfinite(A(:,3)) | ~isfinite(A(:,4)) ...
    | ~isfinite(A(:,5)) | ~isfinite(A(:,6)) | ~isfinite(A(:,7)) ...
    | ~isfinite(A(:,8)) | ~isfinite(A(:,9)));
A(ind,:) = [];

% select CA, OR, WA, NJ, NY counties
ind = find((A(:,1)>=6000 & A(:,1)<=6999) ...  %CA
  | (A(:,1)>=53000 & A(:,1)<=53999)); %...        %WA
%  | (A(:,1)>=34000 & A(:,1)<=34999) ...        %NJ  
%  | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
%  | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
A = A(ind,:);

[n,dim] = size(A);

% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

% % select max subset of data with equal numbers of dem and gop counties
% ngop = length(igop);
% ndem = length(idem);
% if ngop > ndem
%     rgop = randperm(ngop,ndem);
%     Adem = A(idem,:);
%     Agop = A(igop(rgop),:);
%     A = [Adem;Agop];
% else
%     rdem = randperm(ndem,ngop);
%     Agop = A(igop,:);
%     Adem = A(idem(rdem),:);
%     A = [Adem;Agop];
% end  
% [n,dim] = size(A);
% idem = find(A(:,2) >= A(:,3));
% igop = find(A(:,2) < A(:,3));
% num = A(:,2)+A(:,3);
% label = zeros(n,1);
% label(idem) = -1;
% label(igop) = 1;

% set up matrix data and rescale it to [0,1]

X = [A(:,4:9),log(num)];
X(:,1) = X(:,1)/1e4;
% select three data types that distinguish dem and gop counties the most
i1 = 1; % Median Income
i2 = 7; % log(# votes)
i3 = 5; % Bachelor Rate
XX = X(:,[i1,i2,i3]); % data matrix
% rescale all data to [0,1]
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
X1 = (XX(:,1)-xmin)/(xmax-xmin);
X2 = (XX(:,2)-ymin)/(ymax-ymin);
X3 = (XX(:,3)-zmin)/(zmax-zmin);
XX = [X1,X2,X3];

%% Set up optimization problems
% constraint
num_data = length(label);


A11 = diag(label) * [XX ones(num_data,1)];
A12 = eye(num_data);
A21 = zeros(num_data,4);
A = [A11 A12;A21 A12];
b = [ones(num_data,1);zeros(num_data,1)];

% find initial guess
[w,l,lcomp] = FindInitGuess(ones(4,1), A11, ones(num_data,1));
x0 = [w;lcomp];

% run active set method
W = [];
C = 1e3;
gfun = @(x) gfun_soft(x,C);
Hfun = @(x) Hfun_soft(x,C);
[xiter,lm] = ASM(x0,gfun,Hfun,A,b,W);
w = xiter(1:4,end);

% plot of the separating plane
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
%%
figure;
hold on; grid;
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%title('Separating plane for CA data')

p = patch(isosurface(xx,yy,zz,plane,1));
p.FaceColor = 'green';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);
end
%%

function f = fun_soft(x,C)
% w = 4 x 1 column vec
% lam = n x 1 column vec
% x = [w;lam]
w = x(1:4,1);
lam = x(5:end,1);
f = w' * w./2 + C .* sum(lam);
end

function g = gfun_soft(x,C)
w = x(1:4);
lam = x(5:end);
l = length(lam);
g = [w;C.*ones(l,1)];
end

function Hv = Hfun_soft(x,C)
lam = x(5:end);

l = length(lam);
Hv = zeros(4+l);
for j = 1:3
    Hv(j,j) = 1;
end
end