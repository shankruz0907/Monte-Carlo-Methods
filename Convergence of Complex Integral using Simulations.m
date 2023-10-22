clear;
%% Estimate Convergence 
N = 100000000;
F_sum = zeros(N,1);
G_sum = zeros(N,1);
F_mean = zeros(N,1);
G_mean = zeros(N,1);
F_variance = zeros(N,1);
for j = 1:N
    if j==1
        x = rand(1);
        y = rand(1);
        F = exp((x + y)^4);
        G = (exp((x+y)^4))^2;
        F_sum(j) = F;
        G_sum(j) = G;
        F_mean(j) = F_sum(j)/j;
        G_mean(j) = G_sum(j)/j;
        F_variance(j) = G_mean(j) - (F_mean(j))^2;
    else
        x = rand(1);
        y = rand(1);
        F = exp((x + y)^4);
        G = (exp((x+y)^4))^2;
        F_sum(j) = F_sum(j-1) + F;
        G_sum(j) = G_sum(j-1) + G;
        F_mean(j) = F_sum(j)/j;
        G_mean(j) = G_sum(j)/j;
        F_variance(j) = G_mean(j) - (F_mean(j))^2;
    end
end
fprintf('Characteristics of the Estimate:')
fprintf('The estimate of the integral converges to: %1.6f\n',F_mean(N));
fprintf('Standard Deviation of the estimate converges to: %1.6f\n', std(F_mean));
fprintf('Variance of the estimate converges to: %1.6f\n', var(F_mean));
figure(1);
plot(F_mean);
title('Conergence of the estimate');
figure(2);
semilogx(F_mean);
title('Convergence with x-axis - log-scale');
figure(3);
plot(F_variance);
title("Convergence of the variable's variance");
figure(4);
histogram(F_mean, 1000);
title('Distribution of the variable');
fprintf('Characteristics of the Variable:')
fprintf('Varince of the variable X = exp(x+y)^4 with %d iterations is: %1.6f\n', N, F_variance(N));
fprintf('Standard Deviation of the variable X = exp(x+y)^4 with %d iterations is: %1.6f\n', N, sqrt(F_variance(N)));

%% Problem Simplification
%  
% $$ f = \int_{0}^{1}\int_{0}^{1} e^{(x+y)^4} dx dy $$
% 
% $$ => E[e^{(x+y)^4}]\hspace{0.1cm} \forall \hspace{0.1cm}x,y \in [0,1] $$
%
% $$ Defined \hspace{0.1cm} X_i = e^{(x+y)^4}, and \hspace{0.1cm}taking\hspace{0.1cm} N \hspace{0.1cm}iterations $$
%
% $$ Using \hspace{0.1cm} Law \hspace{0.1cm} of\hspace{0.1cm} Large\hspace{0.1cm} Numbers : $$
%
% $$ \bar{X}_N = \frac{1}{N} \sum_{i=1}^{N} X_i $$
%
% $$ E[\bar{X}_N] =  \frac{1}{N} \sum_{i=1}^{N} E[X_i] $$
%
%%
%% Chebyshev's Inequality 
%
% $$ P(|\bar{X}_N - \mu| \geq \varepsilon) \leq \frac{\sigma^2}{N\varepsilon^2} $$
%
% $$ \sigma^2 = E[X^2] - [E[X]]^2$$
%
% $$ Estimating\hspace{0.1cm} this\hspace{0.1cm} i.e.\hspace{0.1cm} [average \hspace{0.1cm}mean\hspace{0.1cm} (X^2 = (e^{((x+y)^4)^2}))] - [average \hspace{0.1cm}mean \hspace{0.1cm} (X = e^{(x+y)^4})]^2  $$
%
% $$ We\hspace{0.1cm} can \hspace{0.1cm}iterate \hspace{0.1cm} \sigma^2 \hspace{0.1cm}for\hspace{0.1cm}N, \hspace{0.1cm}and \hspace{0.1cm}obtain \hspace{0.1cm}the \hspace{0.1cm}variable's \hspace{0.1cm}variance $$
%
%%
fprintf('Then P(|X_N-bar-μ| ≥ ε) ≤ %1.6f/ε^2\n',F_variance(N)/N);
fprintf('Putting ε = %1.6f\n', F_variance(N)/N);
fprintf('This means that the probability of the difference of estimate and μ being greater than 1 Standard Deviation is: %1.6f\n', N/F_variance(N));
%%
%% Confidence Interval 
%
% $$ For\hspace{0.1cm} a \hspace{0.1cm}two-sided\hspace{0.1cm} 95\% \hspace{0.1cm}confidence \hspace{0.1cm}interval,\hspace{0.1cm} we\hspace{0.1cm} use \hspace{0.1cm}\alpha = 0.05.$$
% 
% $$ Then\hspace{0.1cm} calculating \hspace{0.1cm}the
% \hspace{0.1cm}z-statistic \hspace{0.1cm}for \hspace{0.1cm}\pm (\frac{\alpha}{2}), \hspace{0.1cm}which \hspace{0.1cm}is \hspace{0.1cm}
% Z_{0.025} = 1.96,\hspace{0.1cm} Z_{-0.025} = -1.96.$$
% 
% $$ Then \hspace{0.1cm}the \hspace{0.1cm}confidence \hspace{0.1cm}interval \hspace{0.1cm}becomes \hspace{0.1cm}\mu \pm Z_{0.025}*\frac{\sigma}{\sqrt{N}} $$
% 
%%
fprintf('The 2-sided 95%% confidence interval of the estimate is: ± %1.6f from the actual mean\n',1.96*sqrt(F_variance(N)/N));
