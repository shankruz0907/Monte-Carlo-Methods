%%
% 
% $$ S_{i+1}^l  = S_{i}^l  \Bigl( \bigl( r - \delta - \frac{\sigma^2}{2} \bigr) \Delta t + \sigma \Delta W_{i}^l \Bigr)   $$
% 
% $for \hspace{0.1cm} l = 1,2,....,N$
%
% $$ s^{*}(t_i) = quantile_{(95 + 5 \times \frac{i}{n})}   \Bigl( (S_{i}^l)_{l \leq N}  | (K-S_{i}^l)^{+} =  V(i,S_{i}^l) \Bigr) $$
% 
%
%%

%% CRR Binomial Tree
n = 365; T=1; dt = T/n;
time = 0:1:n-1;
sigma = .2; delta = 0;
r = .05; S0 = 80; K = 100;
N = 100000; 

u = exp(sigma*sqrt(dt));
d = 1/u;
p = (exp(dt*(r-delta))-d)/(u-d);

%% prices on binomial tree
S = zeros(n+1,n+1);
S(1,1)=S0;
for t = 2:n+1
    S(1:t-1,t) = S(1:t-1,t-1)*u;
    S(t,t) = S(t-1,t-1)*d;
end

[P,Delta,S_star] = american_put_dyn_prog(S,K,exp(-r*dt),p,n);
S_st = S_star(1:n,1)';
sst = polyfit(time,S_st,5);
f1 = polyval(sst,time);
figure(1)
plot(time,S_st)
hold on
plot(time,f1,'r--','Color',[0.5 0 0])
xlabel('Time Steps')
ylabel('Binomial Tree EE')
hold on
%% LSMC method
S1 = S0*ones(N,n);
ER = exp((r-delta-.5*sigma^2)*dt);

for t = 2:n
    S1(:,t)=ER*S1(:,t-1).*exp(sigma*randn(N,1)*sqrt(dt));
end

[LSMC_Prc,S_star_1,S_star_2] = LSMC_put_func(S1,K,exp(-r*dt));

S_st1 = S_star_1(1:n,1)';
S_st2 = S_star_2(1:n,1)';

plot(time,S_st1,'*','Color',[0 0 0.8])
xlabel('Time Steps')
ylabel('95th Quantile increasing')
hold off

figure(2)
plot(time,S_st)
hold on
plot(time,f1,'r--','Color',[0.5 0 0])
hold on
plot(time,S_st2,'*','Color',[0 0 0.8])
xlabel('Time Steps')
ylabel('100th Quantile')
hold off

%%
%
% The optimal region to excercise is below the line onto the right side
%
% 
% The 95th percentile border is somewhat higher in the LSCMs region than in the CRR model. 
% When the degrees are extended to 20, the model has trouble making predictions when we use 100th percentile for all the time steps and produces a lot of outliers, therefore using it for such high degrees is not recommended. 
% Even at the bottom percentile, the CRR border is more uniform than the LSCMs'. 
% With the simulation of several potential future events, the LSMC approach is able to better depict the complexity of the underlying asset price. 
% On the other hand, LSMC can be computationally demanding, especially for choices with high-dimensional state spaces or where great precision is desired. 
% The CRR model, however, is computationally easier and could be more
% effective for simpler option configurations.
%
%
%
%%

%% Function for binomial tree

function [P,Delta,S_star] = american_put_dyn_prog(S,K,D,p,m)

n = length(S(1,:))-1;
P = zeros(size(S));
P(:,end) = max(K - S(:,end),0);
S_star = zeros(m+1);

for t = (n+1):-1:2
    P(1:t-1,t-1) = D*(p*P(1:t-1,t)+(1-p)*P(2:t,t));
    EE =  K - S(1:t-1,t-1);
    ind = P(1:t-1,t-1) < EE;
    ind1 = P(1:t-1,t-1) <= EE;
    P(ind,t-1) = EE(ind);
    S_star(t-1) = max(S(ind1,t-1));
end

Delta = (P(1,2)-P(2,2))/(S(1,2)-S(2,2));
P = P(1,1);
end

%% LSMC Function

function [LSMC_Prc,S_star_1,S_star_2] = LSMC_put_func(S,K,D)
[~, n]=size(S);
payoff = max(K-S,0);
S_star_1 = zeros(n);
S_star_2 = zeros(n);
V = zeros(size(S));
V(:,end) = payoff(:,end);
% V1 = zeros(size(S));
% V1(:,end) = payoff(:,end);

for t = n:-1:2
    V(:,t-1) = V(:,t)*D;
    ind = payoff(:,t-1)>0;
    psi = power(S(ind,t-1),0:20);
    [Q,~] = qr(psi,0);
    b = Q'*V(ind,t)*D;
    V(ind,t-1) = max(Q*b,payoff(ind,t-1));
end
LSMC_Prc = mean(V(:,1));
for k = 1:n
    ind1 = V(:,k) == (K-S(:,k));
    S_ind_1 = S(ind1,k);
    q = quantile(S_ind_1,0.95+0.05*(k/n));
    q1 = quantile(S_ind_1,1);
    S_star_1(k) = q;
    S_star_2(k) = q1;
end
% LSMC_Prc1 = mean(V1(:,1));
end

