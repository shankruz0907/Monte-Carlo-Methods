clear;
%% Uniform Sampling for estimation
N = 1000;
n = 100;
i = [30, 50, 70];
tic
U = rand(n,N);
X = tan(pi*(U-.5));
Xsort = sort(X);
EX_i = mean(Xsort.');

fprintf('Uniform Sampling Estimation: n = %d, N = %d\n',n,N);

for j = 1:3
    fprintf('Estimated Value for i = %d: EX_%d = %1.6f\n',i(j),i(j),EX_i(i(j)));
end
toc


%% Ordered Statistics
%
%
% $$ X_{(1)} <= X_{(2)} <= X_{(3)} <= X_{(4)} <= ....... <= X_{(n)}; \hspace{0.1cm} where \hspace{0.1cm} X_i $$ -- iid with CDF $$ F_x(n) $$
%
% $$ F_{X_{(1)}}(x) = 1 - (1-F_X(x))^n $$
%
% $$ F_{X_{(n)}}(x) = (F_X(x))^n $$
%
% $$ f_{(k)}(x) = P( X_{(k)} \in  [ x, x + \varepsilon ]) $$
%
% $$ => \hspace{0.1cm} P( one \hspace{0.1cm}X_i \in [ x, x + \varepsilon], \hspace{0.1cm}exactly\hspace{0.1cm} (k-1)\hspace{0.1cm} other\hspace{0.1cm} X_s < x) $$
%
% $$ => \hspace{0.1cm} \sum_{i=1}^{n} P( X_i \in [x,x+\varepsilon], exactly \hspace{0.1cm} (k-1) \hspace{0.1cm} other \hspace{0.1cm}X <x) $$
%
% $$ => \hspace{0.1cm} n. P(X_1 \in[x, x+ \varepsilon]).P(exactly\hspace{0.1cm}(k-1)\hspace{0.1cm}other\hspace{0.1cm}X<x) $$
%
% $$ f_{(k)}(x) => n.P(X_1 \in [x,x+\varepsilon]).[{{n-1} \choose {k-1}}.P(x<k)^{k-1}.P(x>k)^{n-k}] $$
% 
% $$ => f_{(k)}(x) = n.f(x).{{n-1} \choose {k-1}}.(F(x))^{k-1}((1-F(x)))^{n-k} $$
%
% $$ Visibly,\hspace{0.1cm} this\hspace{0.1cm} equation\hspace{0.1cm} is\hspace{0.1cm} of\hspace{0.1cm} the\hspace{0.1cm} form \hspace{0.1cm} cz^{a-1}(1-z)^{b-1} for z \in (0,1), \hspace{0.1cm}where\hspace{0.1cm} c\hspace{0.1cm} is\hspace{0.1cm} the\hspace{0.1cm} normalising\hspace{0.1cm} constant. $$
%
%
%%

%% Using Beta sampling for estimation
tic
fprintf('Uniform Sampling Estimation: n = %d, N = %d\n',n,N);

for j = 1:3
    U_i = betarnd(i(j),n+1-i(j),1,N);
    X_i = tan(pi*(U_i-.5));
    fprintf('Estimated Value for i = %d: EX_%d = %1.6f\n',i(j), i(j), mean(X_i));
end
toc


%%
% The advantage of using the beta sampling is that we can just get the work
% done in 1000 samples instead of using 1000 samples for 100 IIDs and
% ordering them. This also reduces the time taken by the beta sampling
% method.
%%