%% Estimation using Randomized Quasi Monte Carlo Method with Matousek Scrambling  
%
%

%% Given Data
beta = [0,0.8];
est = 1000;
dims = 100;
N = 1000;
j = 1;

%% Non-randomized QMC Estimator

u = 1;
theta_nonrqmc = zeros(length(beta),est);

while u<est+1
    for i = 1:length(beta)
        Z_nonrqmc = norminv(net(sobolset(dims+1,"Skip",1000,"Leap",100),N)');
        X_nonrqmc = beta(i)*Z_nonrqmc(1,:) + sqrt(1-beta(i)^2)*Z_nonrqmc(2:end,:);
        X_l_nonrqmc = sqrt(sum(X_nonrqmc.^2));
        theta_nonrqmc(i,u) = mean(cos(X_l_nonrqmc));
    end
    u = u+1;
end
theta_mean_nonrqmc = mean(theta_nonrqmc,2);
SD_mean_nonrqmc = std(theta_nonrqmc,0,2);

%% Standard Monte Carlo Estimators

%%
% $$ \theta =   \mathrm{E}cos(||X||)  $$
% 
% $$ X \in R^{100} $$
%
% $$ X_i = \beta Z_0 + \sqrt{1-\beta^2} Z_i  $$
%
% $Z_0$ is iid normal(0,1) , $\beta \in (-1,1)$
%
% $$ \hat{\theta} = \frac{1}{1000} \sum_{l=1}^{1000} cos(||X^l||)  $$
% 
% where $X^l \in R^{100}$ is a sample
%
%
%
%%


theta_l = zeros(length(beta),est);
while j<est+1
    for i = 1:length(beta)
        Z_i = randn(dims,N);
        Z_0 = randn(1,N);
        X = beta(i)*Z_0 + sqrt(1-beta(i)^2)*Z_i;
        X_l = sqrt(sum(X.^2));
        theta_l(i,j) = mean(cos(X_l));
    end
    j = j+1;
end
theta_mean_MC = mean(theta_l,2);
SD_mean_MC = std(theta_l,0,2);
figure(1)
subplot(2,1,1)
histfit(theta_l(1,:),50)
title('Standard Monte Carlo Estimators beta = 0')
hxl = xline(theta_mean_nonrqmc(1),'-',{'QMC'});
hxl.FontSize = 15;
hxl.LineWidth = 4;
subplot(2,1,2)
histfit(theta_l(2,:),50)
title('Standard Monte Carlo Estimators beta = 0.8')
hxl = xline(theta_mean_nonrqmc(2),'-',{'QMC'});
hxl.FontSize = 15;
hxl.LineWidth = 4;


%% 1-Dimensional Monte Carlo Estimators Using 100 DoF Non-Central Chi-square RV 
%%
%
% $$ \theta =  \mathrm{E}\mathrm{E} \left[ cos \left(\sqrt{(1-\beta^2) \chi_{100}^2 (Z_0)}\right) | Z_0 \right]    $$
%
%%



k = 1;
theta_1D = zeros(length(beta),est);
while k<est+1
    for i = 1:length(beta)
        Z_0 = randn(1,N);
        Z_chi = ncx2rnd(dims,dims*((beta(i)^2)*(Z_0.^2)/(1-beta(i)^2)));
        X_chi = sqrt((1-beta(i)^2)*Z_chi);
        theta_1D(i,k) = mean(cos(X_chi));
    end
    k = k+1;
end
theta_mean_1D = mean(theta_1D,2);
SD_mean_1D = std(theta_1D,0,2);

figure(2)
subplot(2,1,1)
histfit(theta_1D(1,:),50)
title('1-D Monte Carlo Estimators beta = 0')
hxl = xline(theta_mean_nonrqmc(1),'-',{'QMC'});
hxl.FontSize = 15;
hxl.LineWidth = 4;
subplot(2,1,2)
histfit(theta_1D(2,:),50)
title('1-D Monte Carlo Estimators beta = 0.8')
hxl = xline(theta_mean_nonrqmc(2),'-',{'QMC'});
hxl.FontSize = 15;
hxl.LineWidth = 4;


%% Randomize Quasi-Monte Carlo Estimators

l = 1;
theta_rqmc = zeros(length(beta),est);

while l<est+1
    for i = 1:length(beta)
        Z_rqmc = norminv(net(scramble(sobolset(dims+1,"Skip",1000,"Leap",100),'MatousekAffineOwen'),N)');
        X_rqmc = beta(i)*Z_rqmc(1,:) + sqrt(1-beta(i)^2)*Z_rqmc(2:end,:);
        X_l_rqmc = sqrt(sum(X_rqmc.^2));
        theta_rqmc(i,l) = mean(cos(X_l_rqmc));
    end
    l = l+1;
end
theta_mean_rqmc = mean(theta_rqmc,2);
SD_mean_rqmc = std(theta_rqmc,0,2);

figure(3)
subplot(2,1,1)
histfit(theta_rqmc(1,:),50)
title('RQMC Estimators beta = 0')
hxl = xline(theta_mean_nonrqmc(1),'-',{'QMC'});
hxl.FontSize = 15;
hxl.LineWidth = 4;
subplot(2,1,2)
histfit(theta_rqmc(2,:),50)
title('RQMC Estimators beta = 0.8')
hxl = xline(theta_mean_nonrqmc(2),'-',{'QMC'});
hxl.FontSize = 15;
hxl.LineWidth = 4;



%% Standard Errors Table

VarNames = {'Beta', 'Standard Monte Carlo', '1-Dimensional MC', 'RQMC'};
Errors = table(beta',SD_mean_MC,SD_mean_1D,SD_mean_rqmc,'VariableNames',VarNames);
Standard_Errors_Table = table(Errors,'VariableNames',"Standard Errors for different methods for both betas");
disp(Standard_Errors_Table);

%%
% 
% - For non-randomized QMC, we do not use the scrambling on the SOBOL set as the randomization is brought into the picture by the scrambling. 
% The SOBOL set generates the same lattice regardless of the number of
% times it is run, which means that it is not random and just a process to
% create a mesh.
%
% - From the observations it can be claimed that RQMC estimators are not unbiased, whereas Monte Carlo and 1-D Monte Carlo estimators are unbiased. 
% 
% - The reasoning behind this observation is the bias-inducing Matousek scrambling method in RQMC, which prevents the creation of an unbiased estimator. 
% Since the scramble generates different samples from the same lattice,
% this scramble creates a non-apparent bias.
%
% - The variance-bias tradeoff supports this claim and RQMC estimators would have the least standard deviation. 
% 
% - The claim is confirmed by the results and it can be seen that RQMC has the lowest variance.
%
%
%
%
%
%
%
%
%
%%
