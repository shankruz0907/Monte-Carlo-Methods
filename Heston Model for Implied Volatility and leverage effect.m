%% Heston Model for Implied Volatility
%
% Given CIR process :
%
% $$ dX_t = \kappa(\theta - X_t)dt + \sigma\sqrt{X_t}dW_t $$
%
% conditions : 
%
% $$ X_0 = 0 ; $$
% $$ \kappa = 1 ; $$
% $$ \theta = 0.5 ; $$
% $$ \sigma = \sqrt{2\theta\kappa} ; $$
% $$ T=10 ; $$
% $$ \Delta t = \frac{1}{T \times 365} : $$
%
%
%%

%% CIRProcess: dX=k*(theta−X)*dt+sigma*sqrt(X)*dW
X0=.2;
S0=100;
k=3;
theta=.2;
sigma=sqrt(2*theta*k)*[.35,.75,1];
T = 3/12;
r=.05;
K=(90:1:120)';
rho=[-.2,0,.2];
x = 1;
M=200;
moneyness_lg = log(K*exp(-r*T)/S0);
dt = 1/365;
n=ceil(T/dt);
D = exp(-r*T);
s = "(sigma,rho)=";

for i = 1:length(sigma)
    for j = 1:length(rho)
        Price = Call_price(0.05, 100, 3/12, K, .2, .2, 3, sigma(i), rho(j));
        volatility_imp = blsimpv(S0,K,r,T,Price);
        stock_price = stp_lg(20000, 365, 100, 0.05, 0.2, 3, sigma(i), rho(j), 1/365, r);
        S_T = stock_price(:,n);
        [Sv, Kv] = meshgrid(S_T',K);
        C = D*max(Sv-Kv,0);
        c = mean(C, 2);
        volatility_imp1 = blsimpv(S0,K,r,T,c);
        s1 = "(" + num2str(sigma(i)) + ",";
        s2 = num2str(rho(j)) + ")";
        s3 = strcat(s,s1,s2);
        figure(1)
        subplot(3,3,x)
        plot(moneyness_lg, volatility_imp,"LineWidth",1.0)
        title(s3)
        figure(2)
        subplot(3,3,x)
        plot(moneyness_lg, volatility_imp1,"LineWidth",1.0)
        title(s3)
        x = x+1;
    end
end


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
%%
%
% Feller condition
%
% $$ \sigma^2 \leq 2\theta\kappa $$
%
% $$ X_{i+1} = (1 - \kappa\Delta t) X_i + + \kappa \theta \Delta t + \sigma \sqrt{{X_i}^+} \Delta W_i $$
%
% $$ X_{i+1} = (1-\kappa \Delta t) Xi + (\kappa\theta - \frac{\sigma^2}{2}) \Delta t + \sigma \sqrt{X_{i+1}} \Delta W_i $$ 
%
%
%%

%% Call Price
function [C] = Call_price(r, S0, T, K, X0, theta, k, sigma, rho) 
setdate = datetime(2017,6,29);
maturity = datemnth(setdate, 12*T);

C = optByHestonNI(r, S0, setdate, maturity, 'call', K, X0, theta, k, sigma, rho, 'DividendYield', 0);

end

%%
%
% Sigma tends to enhance the smile's curve, making it steeper. 
% This implies that, in comparison to options with at-the-money strikes, the implied volatility for options with extreme strikes increases.
% The distribution of future stock values has a higher kurtosis when sigma is higher, increasing the likelihood of extreme occurrences, both on the upside and downside. 
% Because of this, implied volatilities for options with extreme strikes are greater. 
% The smile's curve flattens when the vol-of-vol parameter sigma decreases, indicating that the implied volatility for options with extreme strikes is lower than for those at the money.
% This can be seen in the graph,  as we move downwards in each column we can evidence that the curvature of the smile increases(from 0.35 to 1).
%
%
% The volatility smile becomes more upward-sloping as the rho parameter is raised.
% This is due to the fact that greater rho values indicate a stronger positive connection between the price of the asset and the volatility process.
% As implied volatility tends to be higher when the asset price is higher, the volatility smile has a steeper slope when the asset price is higher. 
% On the other hand, a reduction in the rho parameter tends to make the volatility smile more downward-sloping. 
% This is due to the fact that lower rho values suggest a weaker or negative correlation between the asset price and the volatility process. 
% Because of this, the implied volatility tends to be lower at higher asset prices, creating a flatter slope for the volatility smile.
% This can also be seen in the graph, as we move across the rows, we can evidence the smile moves from downward sloping(-0.2) to smile(0) to upward sloping(0.2)
%
%
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