% Specify softmax classifier
PATCH_SIZE = 28; % 28x28 image patches
INPUT_SIZE = PATCH_SIZE * PATCH_SIZE;
NUM_CLASSES = 10;
clear net
net.layer(2).func = 'sigmoid';
net.layer(2).size = 50;
net.layer(2).reg = 'L2';
net.layer(3).func = 'sigmoid';
net.layer(3).size = 30;
net.layer(3).reg = 'L2';
net.layer(4).func = 'softmax';
net.layer(4).size = NUM_CLASSES;
net.layer(4).reg = 'L2';
net.loss = 'logistic';
net.lambda = 1e-4; % Weight regularization parameter

% Inputs and outputs
X = loadMNISTImages('data/train-images.idx3-ubyte');
y = loadMNISTLabels('data/train-labels.idx1-ubyte');
y(y == 0) = 10; % Remap 0 to 10

% Initialize network
[net theta] = initNet(net, X, y);

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

% Train network
options.Method = 'lbfgs'; % Optimisation function
options.maxIter = 400; % Maximum number of iterations
options.display = 'on';
[optTheta, cost] = minFunc(@(p) runNetwork(net, X, y, p), theta, options);

% Test network
XTest = loadMNISTImages('data/t10k-images.idx3-ubyte');
yTest = loadMNISTLabels('data/t10k-labels.idx1-ubyte');
yTest(yTest == 0) = 10; % Remap 0 to 10
[~, ~, h] = runNetwork(net, XTest, yTest, optTheta); % TODO Split forward and backward passes
[~ ,pred] = max(h); % Get softmax predictions assuming labels start at 1
acc = mean(yTest(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);