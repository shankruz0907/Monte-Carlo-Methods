

%%
% 
% We investigate how two simulation techniques for the aforementioned implicit scheme behave ergodically as the parameter varies. 
% The two simulation methods are the implicit scheme approach and the Markov chain transition matrix method. 
% We compare the gamma cumulative distribution function (CDF) of the stationary distribution with the empirical CDFs of the two methods. 
% Then, with $\epsilon$ = 1, we create a Markov chain transition matrix by decomposing the transition distribution of Zt using the generalized Laguerre polynomials as the foundation. 
% As the state space for the Markov chain, 1000 independent points from the gamma density with shape + 1 and scale 1 are used. 
% In order to compare it to the gamma CDF of the stationary distribution, we depict the empirical CDF of the sample route.
%
%
% In part 2, we consider the implicit scheme method, where we generate a 
% sample path of length 104 from the implicit scheme for the CIR process 
% with an initial draw $Z_0$ ∼ gamma(α + 1, 1). We plot the empirical CDF of 
% the sample path and compare it with the gamma CDF of the stationary 
% distribution. Repeat the same process for ϵ = 10^-3. 
% 

%%
% 
% CIR process
% 
% $$ dZ_t = \frac{1}{\epsilon} (1 + \alpha - Z_t)dt + \sqrt{\frac{2Z_t}{\epsilon}} dW_t $$
% 
% $$ \psi_k (z) = \frac{z^{-\alpha} e^{z}}{k!} \frac{d^k}{dz^k} (e^{-z}z^{k + \alpha})  $$
%
% $$ for \hspace{0.1cm} k = 0,1,2... $$
% 
% $$ \mu (z) = \frac{i}{\Gamma (\alpha + 1)} z^{\alpha} e^{-z} $$
% 
% 
% $$ p_t(z|z_0) = \sum_{k=0}^{\inf} c_k e^{\frac{-kt}{\epsilon}} \psi_k (z_0) \psi_k(z) \mu (z) $$
%
% $$ c_k = \frac{ k! \Gamma ( \alpha + 1) }{ \Gamma (k+ \alpha + 1) } = \frac{k}{k+\alpha} c_{k-1} $$
%
% $$ c_0 = 1 $$
%
% $$ \Delta t = \frac{1}{252} \epsilon = 1 $$
% 
% $$ \alpha = 4 $$ 
%
% $$ P_{l,l'} \propto \sum_{k=0}^{20} c_k e^{\frac{-k \Delta t}{\epsilon}} \psi(k)(z^l) \psi_k(z^{l'}) $$ 
%
% $$ \sum_{l' =1}^{ 1000} P_{l,l'} = 1   $$
% 
% 
% 
% 
% 
% 
% 
% 
%%



%% Approach
alpha = 4;
epsilons = [1,10^-3];
k = 20;
n = 1000;
dt =1/252;
z_l = gamrnd(1+alpha,1,1000,1);
c0 = 1;
c_k = zeros(k+1,1);
c_k(1) = c0;
c_knet = zeros(k+1,1);
prod = zeros(1000,1000,20);
for  j = 1:length(epsilons)
    epsilon = epsilons(j);
    for i = 1:k
    
        c_k(i+1) = ((i)*c_k(i))/((i)+ alpha);
    end
    
    for i = 1:k+1
        c_knet(i) = c_k(i)* exp(-((i-1)*dt)/epsilon);
        psi1 = gLaguerre(i-1,z_l,alpha);
        psit = gLaguerre(i-1,z_l',alpha);
        prod(:,:,i) = c_knet(i)*psi1*psit;
    end
    prod_sum = sum(prod,3);
    threshold_p = max(prod_sum,0);
    scaled_sum  = threshold_p./sum(threshold_p,1);
    samp = dtmc(scaled_sum);
    z0_unif = randsample(z_l,1);
    s = simulate(samp,10000);
    sim= s(2:end);
    [cnt_uniqu,uniq] = hist(sim,unique(sim));
    cnts = [uniq,cnt_uniqu'];
    
    
    cnts(:,1) = z_l(:);
    cnts_sort = sortrows(cnts);
    
    cdff = zeros(1000,1);
    for i = 1:1000
        cdff(i) = sum(cnts_sort(1:i,2));
    end
    cdf_plott = cdff./10000;
    
%% 
% $$ \hat F (Z_i) = \frac{\#{j|Z_j \leq Z_i}}{10^4}  $$
%
% $$ Z_{i+1} = Z_i + (\alpha - Z_{i+1})\frac{\Delta t}{\epsilon} + \sqrt{2Z_{i+1}} \frac{\Delta W_i}{\sqrt{\epsilon}} $$
%
% 
% $$ \epsilon = 10^{-3}    $$
%
%
%%    
    
    %% Part B
    
    z_0 = gamrnd(alpha+1,1,1);
    z = zeros(10000,1);
    z_cum = zeros(1000,1);
    z(1) = z_0;
    for i = 2:10000
        dw = randn(1)*sqrt(dt);
        z(i) = (sqrt(2/epsilon)*dw + sqrt((2*(dw)^2/epsilon) + 4*(1 + (dt/epsilon))*(z(i-1)+(alpha*dt/epsilon))))^2/(2*(1+(dt/epsilon)))^2;
    end
    
    [cnt1,unq1] = discretize(sort(z),1000);
    unq1 = unq1(2:end);
    for i  = 1:1000
        z_cum(i) = length(z(z<unq1(i)))/10000;
    end
    figure (j)
    plot(cnts_sort(:,1),cdf_plott)
    hold on
    plot(0:max(z_l),gamcdf(0:max(z_l),alpha+1,1))
    plot(unq1,z_cum)
    legend('Markov Chain', 'Gamma', 'Implicit Scheme')
    title_str = "Epsilon =" + num2str(epsilon);
    title(title_str);
    hold off
end

%% Laguerre Function

function L = gLaguerre(n,z,a)
    syms x
    F = exp(x)*power(x,-a)*diff(exp(-x)*power(x,n+a),n)/factorial(n);
    C = sym2poly(F);
    L = polyval(C,z);
end

%% Answers
% 
%  For epsilon = 1, fast convergence for all of the process is observed, due to
%  which all of the schemes follow closely to each other and hence the
%  curves are similar. When we decrease the value of the epsilon to 10^-3,
%  convergence for the implicit scheme is very slow whereas the markov
%  chain converges faster and hence it closely follows the gamma cdf and
%  implicit scheme is rather irregular.
%  
% 
