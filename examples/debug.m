% Simple network for debugging gradients
nn = NeuralNetwork();
nn.addLayer('input', struct('Function', 'relu', 'Size', 50))
nn.addLayer('output', struct('Function', 'tanh', 'Size', 25))
nn.connect('input', 'output')
nn.Loss = 'squared'; % TODO Make loss layers

% Initialize network
fprintf('Initializing network\n');
theta = nn.initialize();

% Create data
X = rand(50, 100);
y = ones(25, 100);

% Check derivatives
fprintf('Checking gradient numerically\n');
[~, grad] = nn.backProp(X, y);
numgrad = computeNumGrad(@(p)nn.train(p, X, y), theta);
diff = norm(numgrad - grad)/norm(numgrad + grad);
if (diff >= 1e-7)
  disp([grad numgrad]);
  disp(diff);
  fprintf('Norm of the difference between numerical and analytical gradient should be < 1e-7\n');
  return
end