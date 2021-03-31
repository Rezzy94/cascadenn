function [f] = ga_fitness(P,T,nh,W,net)
%Keterangan:
%       nh : Banyak neuron hidden layer
%       m  : Banyaknya neuron input (prediktor)
%       o  : Banyaknya neuron Output

[m n] = size(P);
o = size(T,1);

%Penyusunan Bobot FFNN
%Bobot Input
k=0;
for i=1:nh
    for j=1:m
        k=k+1;
    Wi1(i,j)=W(k);
    end
end
%Bobot Input ke Output
for i=1:o
    for j=1:m
    k=k+1;
    Wi2(i,j)=W(k);
    end
end
%Bobot Bias Input
for i=1:nh
    k=k+1;
    Wbi(i,1)=W(k);
end

%Bobot Output
for i=1:o
    for j=1:nh
    k=k+1;
    Wo(i,j)=W(k);
    end
end

%Bobot Bias Output
for i=1:o
    k=k+1; 
    Wbo(i,1)=W(k);
end


%Replace bobot
net.IW{1,1}=Wi1;
net.IW{2,1}=Wi2;
net.LW{2,1}=Wo;
net.b{1,1}=Wbi;
net.b{2,1}=Wbo;

%Menghitung MSE
yhat = net(P);
error = yhat-T;
f=mse(error);




