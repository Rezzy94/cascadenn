tic
clc
close all
clear all

%============================Training====================================%
%Input Data
data=xlsread('data SO2.xlsx');

% Create Lag Matrix Data
nlag = 1; %maximum lag
y = data;
x = lagmatrix(y,1:nlag);
 
i1 = find(isnan(x)); 
i2 = find(isnan(diff([x ; zeros(1,size(x,2))]) .* x)); 
if (length(i1) ~= length(i2)) || any(i1 - i2) 
 error('Series cannot contain NaN).') 
end 
if any(sum(isnan(x)) == size(x,1)) 
 error('A realization of ''x'' is completely missing (all NaN''s).') 
end 
first_Row = max(sum(isnan(x))) + 1; 
x = x(first_Row:end,:); 
y = y(first_Row:end,:);
Data = [y x]; % Full VAR

%In Sample Data
p = 0.6; %Percentage of Training Data
n = size(Data,1);
n_in = round(n*p);
P_in = x(1:n_in,:)';
T_in = y(1:n_in,:)';
P_out = x(n_in+1:n,:)';
T_out = y(n_in+1:n,:)';

%Preprocessing Data
[pn,ps]=mapstd(P_in);
[tn,ts]=mapstd(T_in);
pout = mapstd('apply',P_out,ps);%Out Sample Data


disp('Train CFNN using Genetic Algorithm and Backpropagation');

% Create Network
n_in = size(P_in,1); %Input Number of In Sample Data
o_in = size(T_in,1); %Target Number of In Sample Data
nh=5; % Number of Neuron in Hiden Layer
net = cascadeforwardnet(nh);
net = configure(net,pn,tn);

kk=(nh+o_in)*(n_in+1)+ (nh*o_in); %Number of Weight 

%Step 1: weight initialization using GA
options=gaoptimset('SelectionFcn',{@selectiontournament,10},'CrossoverFraction',0.8,'CrossoverFcn',@crossoversinglepoint,...
'PlotFcns',@gaplotbestf,'TolFun',1e-30,'PopulationSize',50,'Generations',1000,'Display','final');
[W_ga,fval,exitflag,output,population]= ga(@(W) ga_fitness(pn,tn,nh,W,net),kk,options);
toc

%CFNN Weight Preparation
%Input Weight
k=0;
for i=1:nh
    for j=1:n_in
        k=k+1;
    Wi1(i,j)=W_ga(k);
    end
end

%Cascade Weight
for i=1:o_in
    for j=1:n_in
    k=k+1;
    Wi2(i,j)=W_ga(k);
    end
end
%Input Bias Weight
for i=1:nh
    k=k+1;
    Wbi(i,1)=W_ga(k);
end

%Output Weight
for i=1:o_in
    for j=1:nh
    k=k+1;
    Wo(i,j)=W_ga(k);
    end
end

%Output Bias Weight
for i=1:o_in
    k=k+1; 
    Wbo(i,1)=W_ga(k);
end

%Step 2: Updating Weights using Backpropagation Algorithm
net_ga=cascadeforwardnet(nh);
net_ga = configure(net_ga,pn,tn);

% Network Training Parameter Input
net_ga.trainParam.epochs =1000;
net_ga.trainParam.goal = 1e-5;
net_ga.trainParam.lr = 0.01;

%Weight initialization using GA 
net_ga.IW{1,1}	= Wi1;
net_ga.IW{2,1}	= Wi2;
net_ga.b{1,1}	= Wbi;
net_ga.LW{2,1}	= Wo;
net_ga.b{2,1}   = Wbo;

net=train(net_ga,pn,tn); %Last FNN Model

% Network Simulation
yin = net(pn); %in sample
yout = net(pout); %Out Sample

% Postprocessing Data
Yt_in = mapstd('reverse',yin,ts);
Yt_out = mapstd('reverse',yout,ts);

% Network Evaluation 
% Training Data
e_in = T_in-Yt_in;
RMSE_in = sqrt(mse(e_in));
MAE_in = mean(mean(abs(e_in)));
SMAPE_in = mean((mean(abs(e_in)/(abs(T_in)+abs(Yt_in))))*100);

% Testing Data
e_out = T_out-Yt_out;
RMSE_out = sqrt(mse(e_out));
MAE_out = mean(mean(abs(e_out)));
SMAPE_out = mean((mean(abs(e_out)/(abs(T_out)+abs(Yt_out))))*100);

% Save Output
Yt = [Yt_in, Yt_out];
e = [e_in, e_out];
Output = [Data Yt' e'];
RMSE = [RMSE_in RMSE_out];
MAE = [MAE_in MAE_out];
SMAPE = [SMAPE_in SMAPE_out];
Model = net;
Target = [T_in, T_out]';
Predicted = Yt';
Error = e';
uisave({'Data','Model','RMSE','MAE','SMAPE','Output','Predicted','Target','Error'},'VAR_CFNN_GA_BP_Output');
%Trained ANN is ready for the use

%==========================Plot======================================%
%Training Data
figure()
subplot(221)
a1=[1:size(Yt_in,2)];
plot(a1,T_in(1,:)','b-',a1,Yt_in(1,:)','r-')
title('Predicted Vs Target of Training Data in Taichung')
xlabel('Data');ylabel('Target/Predicted');
legend('Target(-)','Predicted(-)');

subplot(222)
a1=[1:size(Yt_in,2)];
plot(a1,T_in(2,:)','b-',a1,Yt_in(2,:)','r-')
title('Predicted Vs Target of Training Data in Taipei')
xlabel('Data');ylabel('Target/Predicted');
legend('Target(-)','Predicted(-)');

subplot(223)
a1=[1:size(Yt_in,2)];
plot(a1,T_in(3,:)','b-',a1,Yt_in(3,:)','r-')
title('Predicted Vs Target of Training Data in Hsinchu')
xlabel('Data');ylabel('Target/Predicted');
legend('Target(-)','Predicted(-)');

subplot(224)
a1=[1:size(Yt_in,2)];
plot(a1,T_in(4,:)','b-',a1,Yt_in(4,:)','r-')
title('Predicted Vs Target of Training Data in Kaohsiung')
xlabel('Data');ylabel('Target/Predicted');
legend('Target(-)','Predicted(-)');

%Testing Data
figure()
subplot(221)
a2=[1:size(Yt_out,2)];
plot(a2,T_out(1,:)','b-',a2,Yt_out(1,:)','r-')
title('Predicted Vs Target of Testing Data in Taichung')
xlabel('Data');ylabel('Target/Predicted');
legend('Target(-)','Predicted(-)');

subplot(222)
a2=[1:size(Yt_out,2)];
plot(a2,T_out(2,:)','b-',a2,Yt_out(2,:)','r-')
title('Predicted Vs Target of Testing Data in Taipei')
xlabel('Data');ylabel('Target/Predicted');
legend('Target(-)','Predicted(-)');

subplot(223)
a2=[1:size(Yt_out,2)];
plot(a2,T_out(3,:)','b-',a2,Yt_out(3,:)','r-')
title('Predicted Vs Target of Testing Data in Hsinchu')
xlabel('Data');ylabel('Target/Predicted');
legend('Target(-)','Predicted(-)');

subplot(224)
a2=[1:size(Yt_out,2)];
plot(a2,T_out(4,:)','b-',a2,Yt_out(4,:)','r-')
title('Predicted Vs Target of Testing Data in Kaohsiung')
xlabel('Data');ylabel('Target/Predicted');
legend('Target(-)','Predicted(-)');

disp('Network Performance');
fprintf('Percentage of Training Data      = %4.0f\n',p*100);
fprintf('Percentage of Testing Data       = %4.0f\n',(1-p)*100);
disp('-----------------------------------------------------------------');
fprintf('%8s','Criteria');
fprintf('%12s','Training','Testing');
fprintf('\n')
disp('-----------------------------------------------------------------');
fprintf('%8s','RMSE');
fprintf('%12.4f',RMSE);
fprintf('\n')
fprintf('%8s','MAE');
fprintf('%12.4f',MAE);
fprintf('\n')
fprintf('%8s','SMAPE');
fprintf('%12.4f',SMAPE);
fprintf('\n')
disp('-----------------------------------------------------------------');
disp('Trained VAR-CFNN-GA-BP is ready for the use');
%Trained ANN net_f is ready for the use
