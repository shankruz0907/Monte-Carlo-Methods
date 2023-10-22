%% Question 3
%
% $$ \frac{dS_t^l}{S_t^l} = (r-\delta_l)dt + \sigma_ldW_t^l $$
%  
% $$ for \hspace{0.1cm} l = 1,2,3... $$
% r = 0.05
% $\delta_l = 0.02$
% $\sigma_l = 0.3$
% 
% $\rho_{ll'} = 0.2$ 
% 
% $$ dW_t^ldW_t^{l'} = \rho_{ll'}dt$$
% 
% $$ \Bigl( \sum_{l=1}^{3} \lambda_T^l S_T^l - K  \Bigr)^{+} $$
% 
% $$ S_0^1 = S_0^2 = S_0^3 = K = 100 \hspace{0.1cm} T = 1$$
% 
% $$ \lambda_t^l = \frac{S_t^l}{S_t^1 + S_t^2 + S_t^3}  $$
% $$  N = 50,000 $$
% 
% $$ \Bigl( \sum_{l=1}^{3} \lambda_T^l S_T^l - K  \Bigr)^{+} $$
% 
%
%
%%



%%
% 
%  In this problem, we are given three asset prices driven by geometric 
% Brownian motions under the risk-neutral measure. Using Monte Carlo
% simulation, we first price a European basket option with a given payoff.
% We then use the Least-Squares Monte Carlo method to price a Bermudan 
% basket option with a more complex payoff that can be exercised at any 
% of the twelve specified times. To price the European basket option,
% we simulate the three asset prices using the given parameters and 
% calculate the capitalization weights. With these simulated asset 
% prices and weights, we can compute the payoff of the basket option 
% at maturity T and take the average across all the simulations to obtain 
% an estimated price.
% For the Bermudan basket option, we use the LSMC method to find the 
% exercise policy and calculate the continuation value at each exercise
% time. We construct a design matrix using powers of the asset prices up 
% to a certain degree d and perform regressions to estimate the 
% continuation value at each time step. With the continuation values, 
% we can then calculate the expected payoff of the option at each exercise
% time and take the maximum between the expected payoff and the 
% continuation value. We repeat this process backwards in time to obtain 
% an estimated price for the option.
% 

%% Defining Variables
r = 0.05;
deltal = 0.02;
sigmal = 0.3;
rholl = 0.2;
S0 = 100;
K = 100;
n = 50000;
T = 1;
m = 365;
dt1 = 1/12;

%% Monte Carlo Price
ST = zeros(n,3);
lamda1 = zeros(n,3);
sum_lambda = zeros(n,1);
payoff_mc = zeros(n,1);
cov_matrix = [1,0.2,0.2; 0.2,1,0.2; 0.2,0.2,1];
mu_matrix = [0,0,0];
for i = 1:n
    val = mvnrnd(mu_matrix,cov_matrix) ;
    dw1 = val .* sqrt(T);
    ST(i,1) = S0*exp((r - deltal- 0.5 * sigmal^2)*T + sigmal*dw1(:,1));
    ST(i,2) = S0*exp((r - deltal- 0.5 * sigmal^2)*T + sigmal*dw1(:,2));
    ST(i,3) = S0*exp((r - deltal- 0.5 * sigmal^2)*T + sigmal*dw1(:,3));
    lamda1(i,1) = ST(i,1)./(ST(i,1) + ST(i,2)+ ST(i,3));
    lamda1(i,2) = ST(i,2)./(ST(i,1) + ST(i,2)+ ST(i,3));
    lamda1(i,3) = ST(i,3)./(ST(i,1) + ST(i,2)+ ST(i,3));
    sum_lambda(i,1) = ST(i,1)* lamda1(i,1) + ST(i,2)* lamda1(i,2) + ST(i,3)* lamda1(i,3);
    payoff_mc =  exp(-r*T).*max(sum_lambda - K,0);
end
MC_price = mean(payoff_mc);
%% LSMC
dt1 = 1/12;
lamda_1 = zeros(n,13);
lamda_2 = zeros(n,13);
lamda_3 = zeros(n,13);
lamda_1(:,1) = 1/3;
lamda_2(:,1) = 1/3;
lamda_3(:,1) = 1/3;
st_1 = zeros(50000,13);
st_2 = zeros(50000,13);
st_3 = zeros(50000,13);
st_1(:,1) = S0;
st_2(:,1) = S0;
st_3(:,1) = S0;
for i = 1:12
    val = mvnrnd(mu_matrix,cov_matrix,n) ;
    dw1 = val .* sqrt(dt1);
    st_1(:,i+1) = st_1(:,i).*exp((r - deltal- 0.5 * sigmal^2)*dt1 + sigmal*dw1(:,1));
    st_2(:,i+1) = st_2(:,i).*exp((r - deltal- 0.5 * sigmal^2)*dt1 + sigmal*dw1(:,2));
    st_3(:,i+1) = st_3(:,i).*exp((r - deltal- 0.5 * sigmal^2)*dt1 + sigmal*dw1(:,3));
    lamda_1(:,i+1) = st_1(:,i+1)./(st_1(:,i+1) + st_2(:,i+1)+ st_3(:,i+1));
    lamda_2(:,i+1) = st_2(:,i+1)./(st_1(:,i+1) + st_2(:,i+1)+ st_3(:,i+1));
    lamda_3(:,i+1) = st_3(:,i+1)./(st_1(:,i+1) + st_2(:,i+1)+ st_3(:,i+1));
end
S_net = zeros(50000,13,3);
S_net(:,:,1) = st_1;
S_net(:,:,2) = st_2;
S_net(:,:,3) = st_3;
lamda_net = zeros(50000,13,3);
lamda_net(:,:,1) = lamda_1;
lamda_net(:,:,2) = lamda_2;
lamda_net(:,:,3) = lamda_3;
comb = squeeze(sum(lamda_net.*S_net,3));
LSMC_adhoc_Price = LSMC_put_func_ad(comb,K,exp(-r*dt1),10, 13);
LSMC_Price = LSMC_put_func(S_net,exp(-r*dt1),K,10,lamda_net,13);   

result = table(MC_price,LSMC_adhoc_Price,LSMC_Price,'VariableNames',{'Monte-Carlo Payoff','Ad-Hoc LSMC','Multi-Index LSMC'});
disp(result);

%% LSMC Function

function LSMC_Prc = LSMC_put_func(S,D,K,d,lamdas,n)
p1 = repmat((0:d)',1,d+1,d+1);
p2 = permute(p1,[3,1,2]); 
p3 = permute(p1,[2,3,1]);
p_net = p1 + p2+ p3;
index = p_net <=d;
index_comb = [p1(index),p2(index), p3(index)];

combined = squeeze(sum(lamdas.*S,3));
payoff = max(combined - K,0);
V = zeros(size(payoff));
V(:,end) = payoff(:,end);


for t = n:-1:2
    V(:,t-1) = V(:,t);
    ind = payoff(:,t-1)>0;
    psi = (power(S(ind,t-1,1),index_comb(:,1)')).*(power(S(ind,t-1,2),index_comb(:,2)')).*(power(S(ind,t-1,3),index_comb(:,3)'));
    if sum(ind)>0
       [Q, ~] = qr(psi,0);
       b = Q'*V(ind,t)*D;
       V(ind,t-1) = max(Q*b,payoff(ind,t-1));
    end
end


LSMC_Prc = mean(V(:,1));
end
%% Ad Hoc Function

function LSMC_Prc_ad = LSMC_put_func_ad(comb,K,D,d,n)
% [~, n,~]=size(comb);
payoff = max(comb-K,0);

V = zeros(size(comb));
V(:,end) = payoff(:,end);

for t = n:-1:2
   V(:,t-1) = V(:,t)*D;
   ind = payoff(:,t-1)>0;
   psi = power(comb(ind,t-1),0:d);
   [Q, ~] = qr(psi,0);
   b = Q'*V(ind,t)*D;
   V(ind,t-1) = max(Q*b,payoff(ind,t-1));
end

LSMC_Prc_ad = mean(V(:,1));
end

%%
% 
%  LSMC price is the call price for the american option where as monte
%  carlo price is the european call option. Since american call
%  option has the luxury of early exercise, the price for LSMC is expected
%  to higher.
%  
% LSMC ad hoc takes product of the lambdas and the stock prices which takes
% the net price lower than the stock price. Taking power of it will still
% be less than that of LSMC which takes power of the stock prices and then
% product of the lambda and the new stock values. So the net value for LSMC
% ad hoc is less than that of LSMC. Hence a lower LSMC ad hoc price as
% compared to LSMC is seen.
%
