% Specify network
INPUT_SIZE = 64;
HIDDEN_SIZE = 25;
net.layer(2).func = 'sigmoid';
net.layer(2).size = HIDDEN_SIZE;
net.layer(2).reg = 'L2';
net.layer(2).rho = 0.01; % Sparsity parameter
net.layer(3).func = 'sigmoid';
net.layer(3).size = INPUT_SIZE;
net.layer(3).reg = 'L2';
net.cost = 'L2Regression';
net.lambda = 0.0001; % Weight regularization parameter
net.beta = 3; % Sparsity penalty

% Inputs and outputs
load('data/IMAGES.mat'); % Load images
patchsize = 8; % 8x8 image patches
m = 10000; % Number of examples
X = zeros(patchsize*patchsize, m); % Input
for n = 1:m % Extract random patches
  x = ceil(rand*(512 - patchsize));
  y = ceil(rand*(512 - patchsize));
  patch = IMAGES(x:x+patchsize-1, y:y+patchsize-1, ceil(rand*10));
  X(:, n) = reshape(patch, [patchsize*patchsize, 1]);
end
% Normalise for sigmoid activation functions
X = bsxfun(@minus, X, mean(X)); % Remove mean
pstd = 3 * std(X(:));
X = max(min(X, pstd), -pstd) / pstd; % Truncate to +/-3 standard deviations and scale to -1 to 1
X = (X + 1) * 0.4 + 0.1; % Rescale from [-1,1] to [0.1,0.9]
y = X; % Target is reconstructing data

% Initialize network
[net theta] = initNet(net, size(X, 1), size(y, 1));

% Check derivatives
if DEBUG
  [cost, grad] = runNetwork(net, X, y, theta);
  numgrad = computeNumGrad(@(p) runNetwork(net, X, y, p), theta);
  diff = norm(numgrad-grad)/norm(numgrad+grad);
  if (diff > 1e-7)
    disp([grad numgrad]);
    disp(diff);
    fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-7)\n\n');
    return
  end
end

% Gradient descent
options.Method = 'lbfgs'; % Optimisation function
options.maxIter = 400; % Maximum number of iterations
options.display = 'on';
[optTheta, cost] = minFunc(@(p) runNetwork(net, X, y, p), theta, options);

W1 = reshape(optTheta(1:HIDDEN_SIZE*INPUT_SIZE), HIDDEN_SIZE, INPUT_SIZE);
display_network(W1', 12);