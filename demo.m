% Sort paths
clear
addpath minFunc/
addpath functions/

% Specify network
net.layer(2).size = 5;
net.layer(2).func = 'sigmoid';
net.layer(3).size = 3;
net.layer(3).func = 'sigmoid';
net.layer(4).size = 5; % TODO Needs to be the same as previous for L2Regression
net.layer(4).func = 'softmax';

% Inputs
X = ones(8)*0.5; % Input image
m = size(X, 2); % Number of examples
y = zeros([5 m]);
y(1, :) = 1 % Target
net.layer(1).a = X; % Input activation
net.layer(1).size = size(X, 1);
% Set and get weights
[net theta] = initNetParams(net);

% Gradient descent
options.Method = 'lbfgs'; % Optimisation function
options.maxIter = 10; % Maximum number of iterations
options.display = 'on';
[optTheta, cost] = minFunc(@(p) runNetwork(net, X, y, p), theta, options);
[cost, ~, h] = runNetwork(net, X, y, optTheta);
h