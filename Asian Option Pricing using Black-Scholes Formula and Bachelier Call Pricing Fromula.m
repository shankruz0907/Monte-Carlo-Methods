%% Pricing an Asian Option
%
% The underlying price has a risk-neutral SDE :
%
% $$ \frac{dS_t}{S_t} = rdt + \sigma dW_t $$
%
%
% The Asian Option with maturity T and strike K has payoff as follows :
%
% $$ \left( \frac{1}{m} \sum_{i=1}^{m} S_{t_i} - K  \right)^{+} $$
%
%%



%% Comparison on Maturity
K = 90:120;

C_mc=MCM_Asian_Call(1,4);
C_approx=LNApprox_BLS(1,4);
C_bach=Bachelier_Approx(1,4);

figure(1)
plot(K,C_mc)
hold on
plot(K,C_approx)
hold on
plot(K,C_bach)
legend("Monte Carlo", "Log Normal", "Bachelier")
title("T = 1 with 4 time steps")
%%
Data = zeros(4,length(K));
Data(1,:) = K(:);
Data(2,:) = C_mc(:).';
Data(3,:) = C_approx(:);
Data(4,:) = C_bach(:);
VarNames = {'Strike', 'Monte Carlo', 'log normal', 'Bachelier'};

LM_Table = table(Data(1,:).',Data(2,:).',Data(3,:).',Data(4,:).', 'VariableNames',VarNames);

Low_Maturity_Table = table(LM_Table,'VariableNames',"T = 1 with 4 time steps 'Prices To Strike' table");
disp(Low_Maturity_Table);

%%
C_mc1=MCM_Asian_Call(5,20);
C_approx1=LNApprox_BLS(5,20);
C_bach1=Bachelier_Approx(5,20);

figure(2)
plot(K,C_mc1)
hold on
plot(K,C_approx1)
hold on
plot(K,C_bach1)
legend("Monte Carlo", "Log Normal", "Bachelier")
title("T = 5 with 20 time steps")
%%
Data1 = zeros(4,length(K));
Data1(1,:) = K(:);
Data1(2,:) = C_mc1(:).';
Data1(3,:) = C_approx1(:);
Data1(4,:) = C_bach1(:);

HM_Table = table(Data1(1,:).',Data1(2,:).',Data1(3,:).',Data1(4,:).', 'VariableNames',VarNames);

High_Maturity_Table = table(HM_Table,'VariableNames',"T = 5 with 20 time steps 'Prices To Strike' table");
disp(High_Maturity_Table);

%% Monte Carlo 
function[C_mc] = MCM_Asian_Call(T,m)
% Given Data
S0 = 100;
sigma = .2;
r = .05;
K = 90:120;
N = 5000;

D = exp(-r*T);
t = linspace(T/m,T,m);

dt = t(2)-t(1);
W = cumsum(randn(m, N),1)*sqrt(dt);

S = S0*exp((r-.5*sigma^2)*repmat(t,N,1)' + sigma*W);
barS = mean(S,1);

[Sv, Kv] = meshgrid(barS,K);

C_mc = D*mean(max(Sv-Kv,0),2);

end
%% Log Normal Approximation in Black-Scholes Formula
function[C_approx] = LNApprox_BLS(T,m)
% Given Data
S0 = 100;
sigma = .2;
r = .05;
K = 90:120;

D = exp(-r*T);
t = linspace(T/m,T,m);

F = zeros(1,m);
for i = 1:m
    F(i) = S0*exp(r*t(i));
end

G = zeros(m,m);
for j = 1:m
    for i = 1:m
        G(i,j) = F(i)*F(j)*exp(min(t(i),t(j))*(sigma^2));
    end
end

M1 = mean(F);
M2 = sum(sum(G))/m^2;

S0_hat = M1/exp(r*T);
sigma_sq_hat = log(M2/M1^2)/T;

d1 = zeros(1,length(K));
d2 = zeros(1,length(K));

C_approx = zeros(1,length(K));
for i = 1:length(K)
    d1(i) = (log(S0_hat/K(i))+(r+sigma_sq_hat^2)*T)/(sqrt(sigma_sq_hat*T));
    d2(i) = d1(i) - sqrt(sigma_sq_hat*T);
    C_approx(i) = S0_hat*normcdf(d1(i), 0, 1) - K(i)*D*normcdf(d2(i),0,1);
end

end
%%
% $$ Bachelier \hspace{0.1cm} call \hspace{0.1cm} formula $$
%
% $$  C = e^{-rt} \left( (F - K) \Phi(Z) + \sigma \sqrt{T}\phi(Z) \right) $$
%
%%

%% Bachelier Approximation
function[C_bach]=Bachelier_Approx(T,m)
% Given Data
S0 = 100;
sigma = .2;
r = .05;
K = 90:120;
N = 5000;

D = exp(-r*T);
t = linspace(T/m,T,m);

dt = t(2)-t(1);
W = cumsum(randn(m, N),1)*sqrt(dt);

S = S0*exp((r-.5*sigma^2)*repmat(t,N,1)' + sigma*W);
barS = mean(S,1);

[F_hat, sigma_hat] = normfit(barS);

Z = zeros(1,length(K));
C_bach = zeros(1,length(K));

for i = 1:length(K)
    Z(i) = (F_hat-K(i))/(sigma_hat);
    C_bach(i) = D*((F_hat-K(i))*normcdf(Z(i),0,1)+sigma_hat*normpdf(Z(i),0,1));
end

end

%%
% For both T=1 and T=5 , Monte Carlo simulation is better in compared with
% Bachelier as it has less error in comparision with Black-Scholes(log-normal).
%
% Also as T increases from 1 to 5, the option price also increases as a
% consequence of longer maturity which can be evidenced by the increase in
% the number of time steps 4(for T=1) 20(for T=5).
%
%
%
%%