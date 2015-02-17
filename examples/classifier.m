% Specify softmax classifier
layers = loadjson('classifier.json')
for l = 1:length(layers)
  layers{l}
end



%{
PATCH_SIZE = 28; % 28x28 image patches
INPUT_SIZE = PATCH_SIZE * PATCH_SIZE;
NUM_CLASSES = 10;
nn = NeuralNetwork();
nn.addLayer('input', struct('Function', 'sigmoid', 'Size', INPUT_SIZE, 'Reg', 'L2'))
nn.addLayer('output', struct('Function', 'softmax', 'Size', NUM_CLASSES, 'Reg', 'L2'))
nn.connect('input', 'output')
nn.Loss = 'logistic'; % TODO Make loss layers
nn.Lambda = 1e-4; % Weight regularization parameter

% Initialize network
fprintf('Initializing network\n');
theta = nn.initialize();

% Load training and test data
X = loadMNISTImages(['data' filesep 'train-images.idx3-ubyte']);
y = loadMNISTLabels(['data' filesep 'train-labels.idx1-ubyte']);
XTest = loadMNISTImages(['data' filesep 't10k-images.idx3-ubyte']);
yTest = loadMNISTLabels(['data' filesep 't10k-labels.idx1-ubyte']);
y(y == 0) = 10; % Remap 0 to 10
yTest(yTest == 0) = 10; % Remap 0 to 10

% Train network
fprintf('Training network\n')
options.display = true;
[optTheta, cost] = gradientDescent(@(p)nn.train(p, X, y), theta, options);
nn.setParams(optTheta);

% Test network
fprintf('Testing network\n')
h = nn.forwardProp(XTest);
[~, p] = max(h); % Get softmax predictions assuming labels start at 1
acc = mean(yTest == p);
fprintf('Accuracy: %0.3f%%\n', acc * 100);
%}