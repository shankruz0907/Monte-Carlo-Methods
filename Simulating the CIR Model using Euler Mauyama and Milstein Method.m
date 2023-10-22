%% Simulating the CIR Model

%% CIR Process: dX=k*(theta-X)*dt+sigma*sqrt(X)*dW

X0 = .05;
k = 1;
theta = .05;
sigma = sqrt(2*theta*k);
T = 10;
dt = 1/(365*T);
n = T/dt;
M = 100;

%% Euler-Maruyama
X = zeros(M,n);
X(:,1) = X0;
count = zeros(M,n);

for i = 1:M
    for t= 2:n
        dW = randn(1)*sqrt(dt);
        X(i,t) = (1-k*dt)*X(i,t-1) + k*theta*dt + sigma*sqrt(max(X(i,t-1),0))*dW;
        if X(i,t)<0
            count(i,t) = count(i,t) + 1;
        end
    end
end

avg_negatives_per_path = sum(count,"all")/M;
fprintf("The average number of times per path when X_i goes negative in Euler-Maruyama Scheme is: %f \n",avg_negatives_per_path);

figure(1)
plot(linspace(0,T,n),X')
title("X_i simulated using Euler-Maruyama Scheme")
xlabel("Time")
ylabel("X_i")

%% Feller Condition Check Graph
sigma_Feller = sigma*[0.25,0.5,0.75,1,1.25,1.5];
X_Feller = zeros(M,n);
X_Feller(:,1) = X0;

for l = 1:length(sigma_Feller)
    count_Feller = zeros(M,n);
    for i = 1:M
        for t= 2:n
            dW_Feller = randn(1)*sqrt(dt);
            X_Feller(i,t) = (1-k*dt)*X_Feller(i,t-1) + k*theta*dt + sigma_Feller(l)*sqrt(max(X_Feller(i,t-1),0))*dW_Feller;
            if X_Feller(i,t)<0
                count_Feller(i,t) = count_Feller(i,t) + 1;
            end
        end
        
    end
    avg_negatives_fel(l) = sum(count_Feller, "all")/M; 
end

figure(2)
plot(sigma_Feller/sigma,avg_negatives_fel)
title("Feller Condition Check Graph")
xlabel("sigma/√(2*theta*k)")
ylabel("Number of negative points for X_i per path")
%% Milstein Method
X1 = zeros(M,n);
X1(:,1) = X0;
count1 = zeros(M,n);
sigma1 = sigma;

for i = 1:M
    for t= 2:n
        dW1 = randn(1)*sqrt(dt);
        X1(i,t) = X1(i,t-1) + k*(theta-X1(i,t-1))*dt + sigma1*sqrt(X1(i,t-1))*dW1 + (0.5*(sigma1^2)*(dW1^2-dt))/2;
        if X1(i,t)<0
            count1(i,t) = count1(i,t)+1; 
        end
    end
end

avg_negatives_per_path_1 = sum(count1,"all")/M;
fprintf("The average number of times per path when X_i goes negative in Milstein Method is: %f \n",avg_negatives_per_path_1);

figure(3)
plot(linspace(0,T,n),X1')
title("X_i simulated using Milstein Method")
xlabel("Time")
ylabel("X_i")

%% Using Root of Xt in the estimation 
X2 = zeros(M,n);
X2(:,1) = X0;
count2 = zeros(M,n);
sigma2 = sigma;
for i = 1:M
    for t= 2:n
        dW2 = randn(1)*sqrt(dt);
        X2(i,t) = X2(i,t-1) + ((0.5*k*theta-0.125*sigma2^2)/sqrt(max(X2(i,t-1),0)))*dt - 0.5*k*sqrt(max(X2(i,t-1),0))*dt + 0.5*sigma2*dW2;
        X2(i,t) = X2(i,t)^2;
        X2(i,t) = (1-k*dt)*X2(i,t-1) + (k*theta - 0.5*sigma2^2)*dt + sigma*sqrt(X2(i,t))*dW2;
        if X2(i,t)<0
            count2(i,t) = count2(i,t) + 1;
        end
    end
end

avg_negatives_per_path_2 = sum(count2,"all")/M;
fprintf("The average number of times per path when X_i goes negative in the Modified Scheme is: %f \n",avg_negatives_per_path_2);

figure(4)
plot(linspace(0,T,n),X2')
title("X_i simulated using Approximation for √(X_I) *ΔW_i")
xlabel("Time")
ylabel("X_i")