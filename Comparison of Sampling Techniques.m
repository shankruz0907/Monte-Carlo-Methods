clear;
N = 100000000;

%% Box Muller Method
tic
X = zeros(N,1);
Y = zeros(N,1);
for j = 1:N
    U1 = rand(1);
    U2 = rand(1);
    X(j) = sqrt(-2*log(U1))*cos(U2*2*pi);
    Y(j) = sqrt(-2*log(U1))*sin(U2*2*pi);
end
subplot(3,2,1);
histfit(X, 100);
title('Box Muller - X-dist');
subplot(3,2,2);
histfit(Y, 100);
title('Box Muller - Y-dist');

fprintf('Box Muller\n');
fprintf('Mean of X is: %1.6f\n', mean(X));
fprintf('Standard Deviation of X is: %1.6f\n', std(X));
fprintf('Mean of Y is: %1.6f\n', mean(Y));
fprintf('Standard Deviation of Y is: %1.6f\n', std(Y));

toc

%% Marsaglia's Polar Method
tic
X = zeros(N,1);
Y = zeros(N,1);
j = 1;
while j<N
    V1 = 2*rand(1)-1;
    V2 = 2*rand(1)-1;
    S = V1^2 + V2^2;
    if S<1
        X(j) = V1*sqrt(-2*log(S)/S);
        Y(j) = V2*sqrt(-2*log(S)/S);
        j = j+1;        
    end    
end
subplot(3,2,3);
histfit(X, 100);
title("Marsaglia's Polar - X-dist");
subplot(3,2,4);
histfit(Y, 100);
title("Marsaglia's Polar - Y-dist");

fprintf("Marsaglia's Polar\n");
fprintf('Mean of X is: %1.6f\n', mean(X));
fprintf('Standard Deviation of X is: %1.6f\n', std(X));
fprintf('Mean of Y is: %1.6f\n', mean(Y));
fprintf('Standard Deviation of Y is: %1.6f\n', std(Y));

toc

%% Rational Approximation -- 
tic

a0=2.50662823884;
a1=-18.61500062529;
a2=41.39119773534;
a3=-25.44106049637;

b0=-8.47351093090;
b1=23.08336743743;
b2=-21.06224101826;
b3=3.13082909833;

c0=0.3374754822726147;
c1=0.9761690190917186;
c2=0.1607979714918209;
c3=0.0276438810333863;
c4=0.0038405729373609;
c5=0.0003951896511919;
c6=0.0000321767881768;
c7=0.0000002888167364;
c8=0.0000003960315187;

x = zeros(N, 1);
for j = 1:N
    u = rand(1);
    y=u-0.5;

    if abs(y)<0.42
        r1 = abs(y);
        r = r1^2;
        x(j) = r1*(a0 + r*(a1+r*(a2+r*a3)))/(1+r*(b0+r*(b1+r*(b2+r*b3))));
    else
        r = u;
        r = min(r,1-r);
        r = log(-log(r));
        x(j) = c0 + r*(c1+r*(c2+r*(c3+r*(c4+r*(c5+r*(c6+r*(c7+r*c8)))))));
    end
    x(j)=x(j)*sign(y);
end
subplot(3,2,5);
histfit(x, 100);
title('Rational Approximation - X-dist');

fprintf("Rational Approximation\n");
fprintf('Mean of X is: %1.6f\n', mean(x));
fprintf('Standard Deviation of X is: %1.6f\n', std(x));

toc

%% Acceptance-Rejection
tic
c = exp(.5);
X = zeros(1,N);
for i = 1:N
    while 1 > 0
        U1 = rand;
        if U1 < .5
            Y = log(2*U1);
        else
            Y = -log(2*(1-U1));
        end
        f = exp(-.5*Y^2)/sqrt(2*pi);
        g = .5*exp(-abs(Y));
        U2 = rand;
        if U2 < f/(c*g)
            X(i) = Y;
            break
        end
    end
end

subplot(3,2,6);
histfit(X,100,"normal")
title('Acceptance-Rejection - X-dist');

fprintf("Acceptance-Rejection\n");
fprintf('Mean of X is: %1.6f\n', mean(X));
fprintf('Standard Deviation of X is: %1.6f\n', std(X));

toc