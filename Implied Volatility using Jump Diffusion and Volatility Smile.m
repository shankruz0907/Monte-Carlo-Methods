%% Implied Volatility using Jump Diffusion
% 
% The Risk-neutral jump-diffusion SDE for the log-price of an asset
% follows: 
% 
% $$ dX_t = \left( r + \nu - \frac{\sigma^2}{2} \right) dt + \sigma dW_t + log(Y_t)dN_t $$
%
% $$ log(Y_t) \in Normal(a,b^2) ; $$
% $$ dN_t \in Poisson(\lambda dt) ; $$
% $$ \nu = - \left( \mathrm{E}Y_t - 1) \right) \lambda ; $$
%
% $$ S_0 = 100 ; $$
% $$ r = 0.05 ; $$
% $$ \sigma = 0.3 ; $$
% $$ T = \frac{1}{12} ; $$
% $$ K = 90,91,92,...,119,120 ; $$
%
% Jump parameters :
%
% $$ b = 0.1 ; $$ 
% $$ \lambda \in \{2,5\} $$
% $$ a \in \{-0.1,0.1\} $$
%
% $$ \Delta t = \frac{1}{4\times365} $$
%
%%

S0 = 100;
r = 0.05;
T = 1/12;
K = (90:1:120)';
b = 0.1;
dt = 1/(4*365);
n = ceil(T/dt);
sigma = 0.3;
M = 20000;
D = exp(-r*T);
lamda = [2,5];
a = [-0.1,0.1];
moneyness_lg = log(K*exp(-r*T)/S0);
x = 1;
s = "(lamda,a) pair is:";

for i = 1:length(lamda)
    for j = 1:length(a)
        v = (1-exp(a(j)+(b^2)/2))*lamda(i);
        S = jd_call(S0,sigma,lamda(i),a(j),b,n,dt,v,r,M);
        S_T = S(:,n);
        [Sv, Kv] = meshgrid(S_T',K);
        Payoff = mean(D*max(Sv-Kv,0),2);
        volatility_imp = blsimpv(S0,K,r,T,Payoff);
        s1 = "(" + num2str(lamda(i)) + ",";
        s2 = num2str(a(j)) + ")";
        s3 = strcat(s,s1,s2);
        figure(1)
        subplot(length(lamda),length(a),x)
        plot(moneyness_lg, volatility_imp,"LineWidth",1.0)
        title(s3)
        xlabel("Log Moneyness")
        ylabel("Implied Volatility")
        sgtitle("Implied Volatility vs Log Moneyness using 20000 simulations")
        for l = 1:length(K)
            Price(l) = jump_diffusion_call(100,.05,1/12,K(l),0.3,lamda(i),a(j),b,v);
        end
        volatility_imp_integrated = blsimpv(S0,K,r,T,Price);
        figure(2)
        subplot(length(lamda),length(a),x)
        plot(moneyness_lg, volatility_imp_integrated,"LineWidth",1.0)
        title(s3)
        xlabel("Log Moneyness")
        ylabel("Implied Volatility")
        sgtitle("Implied Volatility vs Log Moneyness using quadgk integration")
%         figure(3)

        x = x+1;
    end
end


function S = jd_call(S0,sigma,lamda,a,b,n,dt,v,r,M)
X=log(S0)*ones(M,n);
J=ones(M,n);

for t=2:n
        Z=normrnd(0,1,1,M);
        Nt = poissrnd(lamda*dt, M, 1);
        M_Y = Nt*a + b*sqrt(Nt).*normrnd(0,1,M,1);    
        X(:,t)=X(:,t-1)+(r+v-.5*sigma^2)*dt+sigma*sqrt(dt).*Z'+M_Y;
        J(:,t)=J(:,t-1).*exp(M_Y);
end
S=exp(X);
end

function C = jump_diffusion_call(S0,r,T,K,sigma,lambda,a,b,v)
D = exp(-r*T);
k = log(S0./K) + r*T ;

%%%% ITM Integral
phi = @(u) real(exp(1i*u.*k).*exp(1i*u*r*T + 1i*u*v*T - .5*T*(sigma*u).^2 + lambda*T*(exp(1i*u*a-.5*(u*b).^2)-1))./(1i*u));

I1 = D*K.*(.5 + quadgk(phi,0,inf)/pi);

%%%% Delta Integral
phi_negi = exp(1i*(-1i)*r*T +1i*(-1i)*v*T - .5*T*(sigma*(-1i)).^2 + lambda*T*(exp(1i*(-1i)*a-.5*((-1i)*b).^2)-1));

phi = @(u) real(exp(1i*u.*k).*exp(1i*(u-1i)*r*T + 1i*(u-1i)*v*T - .5*T*(sigma*(u-1i)).^2 + lambda*T*(exp(1i*(u-1i)*a-.5*((u-1i)*b).^2)-1))./(1i*u*phi_negi));

I2 = S0*(.5 + quadgk(phi,0,inf)/pi);

%%%% call price
C = I2 - I1;
end


%%
%
% In the jump diffusion model, the parameter lambda controls the frequency of the jumps in the underlying asset price process. 
% This is because higher values of lambda imply more frequent jumps in the asset price process, which can lead to higher volatility and higher implied volatilities for options with extreme strikes.
% As a result, the volatility smile becomes steeper, with higher implied volatilities for out-of-the-money options. 
% This can be seen from the graph above
% 
% 
% A decrease in the lambda parameter lead to lower volatility and lower implied volatilities for options with extreme strikes.
% This is because lower values of lambda imply fewer jumps in the asset price process, which can lead to lower volatility and lower implied volatilities for options with extreme strikes. 
% As a result, the volatility smile becomes flatter, with lower implied volatilities for out-of-the-money options. 
% This can be evidenced from the graph above
%
%
% In the jump diffusion model, the parameter "a" controls the volatility of the continuous diffusion process that governs the underlying asset price.
% An increase in the "a" parameter tends to make the volatility smile more upward-sloping.
% This is because higher values of "a" imply higher volatility for the continuous diffusion process, which can lead to higher implied volatilities for out-the-money options and lower implied volatilities for at-of-the-money options. 
% 
% 
% A decrease in the "a" parameter tends to make the volatility smile more upward-sloping. 
% This is because lower value of "a" implies lower volatility for the continuous diffusion process, which can lead to higher implied volatilities for at-the-money options and lower implied volatilities for out-of-the-money options. 
%
%
%
%
%%
