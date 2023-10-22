%% Asian Option Pricing using Control Variates and Antithetic Sampling
%
% $$ \frac{dS_t}{S_t} = (r-q)dt + \sigma dW_t $$
%
%%


%% Given Data
S0 = 100;
sigma = .2;
r = .05;
q = .02;
T = 1;
m = 4;
K = 90:5:120;
D = exp(-r*T);
t = linspace(T/m,T,m);
N = 100; 
L = 100;
N_anti = 50;

%% Standard Monte Carlo
%%
%
% $$ \left( \frac{1}{m} \sum_{i=1}^{m} S_{t_i} - K  \right)^{+} $$
%
%%


C_mc = zeros(length(K),L);

for ctr = 1:L
    W = cumsum(randn(N,m)*sqrt(T/m),2);
    S = S0*exp((r-q-.5*sigma^2)*t + sigma*W);
    M = mean(S,2);
    [Mv,Kv] = meshgrid(M,K);
    psi_asian = D*max(Mv-Kv,0);
    
    C_mc(:,ctr) = mean(psi_asian,2);
end

%% Control Variates Method
%%
%
% $$ \left(  \left( \prod_{i=1}^{m} S_{t_i} \right)^{\frac{1}{m}} - K \right)^{+} $$
%
%%
C_cv = zeros(length(K),L);

for ctr = 1:L
    W_cv = cumsum(randn(N,m)*sqrt(T/m),2);
    S_cv = S0*exp((r-q-.5*sigma^2)*t + sigma*W_cv);
    S_geo = geomean(S_cv,2);
    [Sv_geo,Kv_geo] = meshgrid(S_geo,K);
    [Sv_asian,Kv_asian] = meshgrid(mean(S_cv,2),K);
    psi_cv_asian = D*max(Sv_asian-Kv_asian,0);
    psi_euro_geo = D*max(Sv_geo-Kv_geo,0);
    for i = 1:m
        M1_it(i) = exp(((r-q-.5*sigma^2)*((T*i)/m^2))+((sigma^2*T*i^2)/(2*m^3)));
        M2_it(i) = exp((2*(r-q-.5*sigma^2)*((T*i)/(m^2)))+((2*sigma^2*T*i^2)/m^3));
    end
    M1 = prod(M1_it);
    F_hat = M1*S0;
    M2 = prod(M2_it);
    sigma_hat = sqrt(log(M2/M1^2)/T);
    Epsi_euro = blkprice(F_hat,K,r-q,T,sigma_hat);
    b = (psi_cv_asian - mean(psi_cv_asian,2))*(psi_euro_geo - Epsi_euro')'/N;
    b = diag(b)./diag((psi_euro_geo -Epsi_euro')*(psi_euro_geo -Epsi_euro')'/N);
    psi_cv = psi_cv_asian - diag(b)*(psi_euro_geo - Epsi_euro');

    C_cv(:,ctr) = mean(psi_cv,2);

end

%% Antithetic Sampling
%%
%
% $$ \tilde{W} = - W $$
%
%%
    
C_antithetic = zeros(length(K),L);
for ctr = 1:L
    W_anti = cumsum(randn(N_anti,m)*sqrt(T/m),2);
    S1 = S0*exp((r-q-.5*sigma^2)*t + sigma*W_anti);
    S2 = S0*exp((r-q-.5*sigma^2)*t - sigma*W_anti);
    M1 = mean(S1,2);
    M2 = mean(S2,2);
    [M1v,Kv1] = meshgrid(M1,K);
    [M2v,Kv2] = meshgrid(M2,K);
    psi1_asian = 0.5*D*max(M1v-Kv1,0);
    psi2_asian = 0.5*D*max(M2v-Kv2,0);
    psi_anti_asian = psi1_asian + psi2_asian;
    C_antithetic(:,ctr) = mean(psi_anti_asian,2);
end


%% Computing mean and standard error of trials
avg = [mean(C_mc,2),mean(C_cv,2),mean(C_antithetic,2)];
err = [std(C_mc,[],2),std(C_cv,[],2),std(C_antithetic,[],2)];

plot(K,avg)
legend('Monte Carlo Prices','Control Variate Prices','Antithetic Prices')
title('Asian Call Option Prices')
xlabel('Strike Price')
ylabel('Option Price')

VarNames = {'Strike Price','Monte Carlo Method', 'Control Variates Method', 'Antithetic Sampling Method'};
Stderrs = table(K',err(:,1),err(:,2),err(:,3),'VariableNames',VarNames);
Errors = table(Stderrs,'VariableNames',"Standard Errors of Asian Option Prices");
disp(Errors);

%%
% - We are observing the lowest variance in the Control Variates(CV) of three
% methods because the auxillary payoff function is strongly correlated with
% the primary payoff function. 
%
% - In addition to this CV are robust to various
% types of payoff functions, which is not the case with antithetic
% sampling. 
%
% - In the case of Antithetic, a high correlation b/w the samples
% would not result in efficient variance reduction, however this is
% opposite in the control variate method as high correlation b/w the arithmetic and geometric payoff
% functions would result in better variance reduction.
%
% - This significant correlation between our arithmetic and geometric mean which causes significant reduction in 
% variance as the reduction is equal  to $(1-\rho^2)$.
% 
% - We know that antithetic causes a reduction in variance equal to  $\frac{1}{N}$ 
% 
% - In this instance N = 100 which reduces variance by 0.01.
% 
% - But $(1-\rho^2)$ is roughly 0.9 which results in a variance reduction of around 0.036, theoritically.
% 
% - The Variance reduction obtained is different because the two random variables are significantly different than those assumed for the
% theoritical operations.
%
%%