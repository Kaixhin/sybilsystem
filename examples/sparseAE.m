% Specify network
PATCH_SIZE = 28; % 28x28 image patches
INPUT_SIZE = PATCH_SIZE * PATCH_SIZE;
HIDDEN_SIZE = 196;
net.layer(2).func = 'sigmoid';
net.layer(2).size = HIDDEN_SIZE;
net.layer(2).reg = 'L2';
net.layer(2).rho = 0.1; % Sparsity parameter
net.layer(3).func = 'sigmoid';
net.layer(3).size = INPUT_SIZE;
net.layer(3).reg = 'L2';
net.cost = 'L2Regression';
net.lambda = 3e-3; % Weight regularization parameter
net.beta = 3; % Sparsity penalty

% Inputs and outputs
IMAGES = loadMNISTImages('data/train-images.idx3-ubyte');
X = IMAGES(:, 1:10000);
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