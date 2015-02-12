% Specify softmax classifier
PATCH_SIZE = 28; % 28x28 image patches
INPUT_SIZE = PATCH_SIZE * PATCH_SIZE;
NUM_CLASSES = 10;
nn = NeuralNetwork();
nn.addLayer('input', struct('Function', 'sigmoid', 'Size', INPUT_SIZE, 'Reg', 'L2'))
nn.addLayer('output', struct('Function', 'softmax', 'Size', NUM_CLASSES, 'Reg', 'L2'))
nn.connect('input', 'output')
nn.Loss = 'logistic';
nn.Lambda = 0; % Weight regularization parameter

% Initialize network
theta = nn.initialize();

% Inputs and outputs
X = loadMNISTImages(['data' filesep 'train-images.idx3-ubyte']);
y = loadMNISTLabels(['data' filesep 'train-labels.idx1-ubyte']);
y(y == 0) = 10; % Remap 0 to 10

% Run network
h = nn.forwardProp(X);
grad = nn.backProp(h, y);