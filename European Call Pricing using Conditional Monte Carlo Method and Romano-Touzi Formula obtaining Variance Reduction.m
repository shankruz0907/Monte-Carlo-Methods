%% European Call Pricing using Conditional Monte Carlo Method and Romano-Touzi Formula obtaining Variance Reduction
%
% $$ \frac{dS_t}{S_t} = rdt + \sqrt{X_t} \left( \rho dB_t + \sqrt{1- \rho^2} dW_t   \right) $$
% 
% $$ dX_t = \kappa(\theta - X_t)dt + \sigma\sqrt{X_t}dB_t $$
%
%
%%
%% Given Data
X0=.2;
S0=100;
k=3;
theta=.2;
sigma=sqrt(2*theta*k);
T = 3/12;
r=.05;
K=(90:5:120)';
rho=[0,-0.3,-0.7];
dt = 1/365;
n=floor(T/dt);
D = exp(-r*T);
M = 20000;
x = 1;

%% Heston Option Pricing

for j = 1:length(rho)
    c_heston(:,j) = Call_price(r, S0, T, K, X0, theta, k, sigma, rho(j));
    volatility_imp_Heston(:,j) = blsimpv(S0,K,r,T,c_heston(:,j));
end

%% Standard Monte Carlo Method
volatility_imp = zeros(length(K),100,length(rho));
c = zeros(length(K),100,length(rho));
while x<101
    for j = 1:length(rho)
        SP_MC = stp_lg(M, n, S0, X0, theta, k, sigma, rho(j), dt, r);
        S_T = SP_MC(:,n);
        [Sv, Kv] = meshgrid(S_T',K);
        C = D*max(Sv-Kv,0);
        c(:,x,j) = mean(C,2);
        volatility_imp(:,x,j) = blsimpv(S0,K,r,T,c(:,x,j));
    end
    x = x+1;
end
C_MC_mean = mean(c,2);
C_MC_sd = std(c,0,2);
Impvol_MC_mean = mean(volatility_imp,2);
Impvol_MC_sd = std(volatility_imp,0,2);


%% Conditional Monte Carlo Method

%%
%
% $$ C^{stoch\hspace{0.1cm} - \hspace{0.1cm} vol}(s,x,T) = \mathrm{E} \left[ C^{bs} (se^Z, \sigma(X), T) | S_0 = s, X_0 = x \right] $$
%
% $C^{bs}(s,x,T)$ denotes the  Black-Scholes  call  price
%
% $$ Z = \rho \int_{0}^{T}\sqrt{X_t}dB_t\ - \frac{\rho^2}{2} \int_{0}^{T}X_tdt\ $$ 
%
% $$ \sigma^2(X) = \frac{1-\rho^2}{T} \int_{0}^{T}X_tdt\  $$
%
%
%
%%



%%
% From the Heston Model,
%
% $$ \frac{dS_t}{S_t} = rdt + \sqrt{X_t} \left( \rho dB_t + \sqrt{1- \rho^2} dW_t   \right) $$
%
% $$ dX_t = \kappa(\theta - X_t)dt + \sigma\sqrt{X_t}dB_t $$  
% 
% =>
% 
% $$ \sigma\sqrt{X_t}dB_t =  dX_t - \kappa(\theta - X_t)dt  $$
%
% $$ \sqrt{X_t}dB_t = \frac{dX_t - \kappa(\theta - X_t)dt}{\sigma} $$
% 
% $$ Z = \frac{\rho}{\sigma} \int_{0}^{T}dX_t\ -  \kappa(\theta - X_t)dt - \frac{\rho^2}{2} \int_{0}^{T}X_tdt\   $$
%
% $$ = \frac{\rho}{\sigma} \left(  \int_{0}^{T}dX_t\ - \int_{0}^{T} \kappa \theta dt\  + \int_{0}^{T} \kappa X_t dt\  \right) - \frac{\rho^2}{2} \int_{0}^{T} X_t dt\ $$
%
% $$ Z = \frac{\rho}{\sigma}   \left(  X_t - X_0 - \int_{0}^{T} \kappa \theta dt\  + \int_{0}^{T} \kappa X_t dt\  \right) -   \frac{\rho^2}{2} \int_{0}^{T} X_t dt\    $$
%
% This equation is used to compute the values of Z which is used below
%
%%



Z = zeros(M,length(rho),100);
sigsq_x = zeros(M,length(rho),100);
S_new = zeros(M,length(rho),100);
x_new = zeros(M,length(rho),100);
C_bs_new = zeros(M,length(rho),length(K),100);
c_stochvol = zeros(length(rho),length(K),100);
volatility_imp_CMC = zeros(length(rho),length(K),100);
y=1;
while y<101
    for j = 1:length(rho)
        [a_p, a_T]= strate(M,n,X0,theta,k,sigma,dt);
        Z(:,j,y) = (rho(j)/sigma)*(a_T - k*theta*T + k*a_p) - 0.5*a_p*rho(j)^2 ;
        sigsq_x(:,j,y) = (1-rho(j)^2)*a_p/T; 
        S_new(:,j,y) = S0*exp(Z(:,j,y));
        x_new(:,j,y) = sqrt(sigsq_x(:,j,y));
        for i = 1:length(K)
            C_bs_new(:,j,i,y) = blsprice(S_new(:,j,y),K(i),r,T,x_new(:,j,y));
        end
        c_stochvol(j,:,y) = mean(C_bs_new(:,j,:,y),1);
        for i = 1:length(K)
            volatility_imp_CMC(j,i,y) = blsimpv(S0,K(i),r,T,c_stochvol(j,i,y));
        end
        
    end
    y = y+1;
end
C_CMC_mean = mean(c_stochvol,3);
C_CMC_sd = std(c_stochvol,0,3);
Impvol_CMC_mean = mean(volatility_imp_CMC,3);
Impvol_CMC_sd = std(volatility_imp_CMC,0,3);


%% Analysis of Prices for rho = 0
VarNames = {'Strike Price','Heston Price', 'Avg MC', 'Avg C-MC', 'std. err. MC', 'std. err. C-MC','err. MC/C-MC'};
Prices = table(K,c_heston(:,1),C_MC_mean(:,1),C_CMC_mean(1,:)',C_MC_sd(:,1),C_CMC_sd(1,:)',C_MC_sd(:,1)./C_CMC_sd(1,:)','VariableNames',VarNames);
disp(Prices);
%% Analysis of Prices for rho = -0.3
Prices1 = table(K,c_heston(:,2),C_MC_mean(:,2),C_CMC_mean(2,:)',C_MC_sd(:,2),C_CMC_sd(2,:)',C_MC_sd(:,2)./C_CMC_sd(2,:)','VariableNames',VarNames);
disp(Prices1);
%% Analysis of Prices for rho = -0.7
Prices2 = table(K,c_heston(:,3),C_MC_mean(:,3),C_CMC_mean(3,:)',C_MC_sd(:,3),C_CMC_sd(3,:)',C_MC_sd(:,3)./C_CMC_sd(3,:)','VariableNames',VarNames);
disp(Prices2);
%% Analysis of Implied Volatilities for rho = 0
VarNames1 = {'Strike Price','Heston Imp.Vol.', 'Avg MC', 'Avg C-MC', 'std. err. MC', 'std. err. C-MC', 'err. MC/C-MC'};
Prices3 = table(K,volatility_imp_Heston(:,1),Impvol_MC_mean(:,1),Impvol_CMC_mean(1,:)',Impvol_MC_sd(:,1),Impvol_CMC_sd(1,:)',Impvol_MC_sd(:,1)./Impvol_CMC_sd(1,:)','VariableNames',VarNames1);
disp(Prices3);
%% Analysis of Implied Volatilities for rho = -0.3
Prices4 = table(K,volatility_imp_Heston(:,2),Impvol_MC_mean(:,2),Impvol_CMC_mean(2,:)',Impvol_MC_sd(:,2),Impvol_CMC_sd(2,:)',Impvol_MC_sd(:,2)./Impvol_CMC_sd(2,:)','VariableNames',VarNames1);
disp(Prices4);
%% Analysis of Implied Volatilities for rho = -0.7
Prices5 = table(K,volatility_imp_Heston(:,3),Impvol_MC_mean(:,3),Impvol_CMC_mean(3,:)',Impvol_MC_sd(:,3),Impvol_CMC_sd(3,:)',Impvol_MC_sd(:,3)./Impvol_CMC_sd(3,:)','VariableNames',VarNames1);
disp(Prices5);

%% Function for short rate
function [A_p, A_T] =  strate(M, n, X0, theta, k, sigma, dt)
A = zeros(M,n);
A(:,1) = X0;
for t= 2:n
    dW = randn(M,1)*sqrt(dt);
    A(:,t) = (1-k*dt)*A(:,t-1) + k*theta*dt + sigma*sqrt(max(A(:,t-1),0)).*dW;
end
A_T = A(:,n) - A(:,1);
A_p = sum(A,2)*dt;
end


%% Function For Simulations

function Y_p =  stp_lg(M, n, S0, X0, theta, k, sigma, rho, dt, r)
X = zeros(M,n);
X(:,1) = X0;
Y = zeros(M,n);
Y(:,1) = log(S0);

for t= 2:n
    dW = randn(M,1)*sqrt(dt);
    dB = randn(M,1)*sqrt(dt);
    X(:,t) = (1-k*dt)*X(:,t-1) + k*theta*dt + sigma*sqrt(max(X(:,t-1),0)).*dW;
    Y(:,t) = Y(:,t-1) + (r-.5*X(:,t-1))*dt + sqrt(max(X(:,t-1),0)).*(rho*dW+sqrt(1-rho^2)*dB);
end
Y_p=exp(Y);
end

%% Call Price Using Heston Pricing Model
function [C] = Call_price(r, S0, T, K, X0, theta, k, sigma, rho) 
setdate = datetime(2017,6,29);
maturity = datemnth(setdate, 12*T);

C = optByHestonNI(r, S0, setdate, maturity, 'call', K, X0, theta, k, sigma, rho, 'DividendYield', 0);
end


%% Theory of Variance Reduction
%
% When $\rho$ is 0 for the SMC, only one random process is incorporated in the
% underlying asset process leading to highest underlying volatility as compared to when $\rho$ is not equal to 0 or 1.
% When $\rho \neq  0 or 1$, the factorisation of two random
% processes effects in a lower underlyin volatility
%
% For SMC $S_0$ does not change as rho changes, therefore  increase in mod rho
% decreases standard error of the option price and implied volatility for
% OTM options, and the converse for ITM options
% 
% FOR CMC, as $\rho$ increases, the underlying volatility always decreases.
% Also as $\rho$ chages $S_0$ changes with a coefficient of $e^Z$, therefore
% variance of $S_0 \geq 0$
% increasing the deviations in the calculations of option prices and imp vols
%
% If $S_0$ would have been constant, there would have been a significant
% reduction in the std devs. Now since $S_0$ is not a constant the "reduced" standard
% deviation is increased due to this variance. Therefore as  $|\rho|$ increases the
% standard error for calculations of option prices and implied volatility decreases
% for both SMC and CMC for OTM options and the reverse holds true for ITM
% options, but the ratio of std errors of SMC/CMC decreases due to the
% aforementioned reasons.
%
%%