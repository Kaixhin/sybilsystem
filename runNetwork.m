function [cost, grad, h] = runNetwork(net, X, y, theta)
  % RUNNETWORK Run a network
  % net:    Network
  % X:      Input
  % y:      Output
  % theta:  Network parameters
  % cost:   Cost
  % grad:   Network parameter gradients with respect to cost
  % h:      Hypothesis
  L = length(net.layer);
  m = size(X, 2);
  
  % Set network parameters
  prevWbLength = 0;
  for l = 2:L
    WLength = net.layer(l).size * net.layer(l-1).size;
    bLength = net.layer(l).size;
    W = theta(1 + prevWbLength:prevWbLength + WLength); % Extract weights
    net.layer(l-1).W = reshape(W, [net.layer(l).size net.layer(l-1).size]); % Set weights    
    net.layer(l-1).b = theta(1 + prevWbLength + WLength:prevWbLength + WLength + bLength); % Extract and set biases
    prevWbLength = prevWbLength + WLength + bLength;
  end
  
  % Forward propagation
  cost = 0;
  net.layer(1).a = X; % Input activation
  for l = 2:L
    net.layer(l).z = net.layer(l-1).W * net.layer(l-1).a + repmat(net.layer(l-1).b, [1 m]); % Weighted inputs + biases
    fn = str2func(net.layer(l).func); % Layer function
    net.layer(l).a = fn(net.layer(l).z); % Activation
    if isfield(net.layer(l), 'reg') && strcmp(net.layer(l).reg, 'L2')
      cost = cost + (net.lambda/2) * sum(sum(net.layer(l-1).W.^2)); % L2 weight regularization cost
    end
    if isfield(net.layer(l), 'rho') && isscalar(net.layer(l).rho)
      rho = net.layer(l).rho; % Sparsity parameter
      rhoHat = (1/m) * sum(net.layer(l).a, 2); % Average activations
      net.layer(l).rhoHat = rhoHat;
      KL = rho*log(rho./rhoHat) + (1 - rho)*log((1 - rho)./(1 - rhoHat)); % Kullback-Leibler divergence
      cost = cost + net.beta * sum(KL); % Sparsity cost
    end
  end
  h = net.layer(L).a; % Hypothesis
  lossFn = str2func(net.loss); % Loss function
  cost = cost + lossFn(h, y); % Loss cost
  
  % Backward propagation
  grad = [];
  lossFnD = str2func(strcat(net.loss, 'D'));
  net.layer(L).delta = lossFnD(h, y); % Loss cost error
  if isfield(net.layer(L), 'rho') && isscalar(net.layer(L).rho)
    rho = net.layer(L).rho; % Sparsity parameter
    rhoHat = net.layer(L).rhoHat; % Average activations
    net.layer(L).delta = net.layer(L).delta + repmat(net.beta*(-(rho./rhoHat) + ((1 - rho)./(1 - rhoHat))), [1 m]); % Sparsity derivative
  end  
  fnD = str2func(strcat(net.layer(l).func, 'D'));
  net.layer(L).delta = net.layer(L).delta .* fnD(net.layer(L).z); % Error due to weighted inputs
  for l = L-1:-1:2
    net.layer(l).delta = net.layer(l).W' * net.layer(l+1).delta; % Error from outputs
    if isfield(net.layer(l), 'rho') && isscalar(net.layer(l).rho)
      rho = net.layer(l).rho; % Sparsity parameter
      rhoHat = net.layer(l).rhoHat; % Average activations
      net.layer(l).delta = net.layer(l).delta + repmat(net.beta*(-(rho./rhoHat) + ((1 - rho)./(1 - rhoHat))), [1 m]); % Sparsity derivative
    end
    fnD = str2func(strcat(net.layer(l).func, 'D'));
    net.layer(l).delta = net.layer(l).delta .* fnD(net.layer(l).z); % Error due to weighted inputs
    net.layer(l).dW = (1/m) * net.layer(l+1).delta * net.layer(l).a'; % Weight derivative
    if isfield(net.layer(l+1), 'reg') && strcmp(net.layer(l+1).reg, 'L2')
      net.layer(l).dW = net.layer(l).dW + net.lambda * net.layer(l).W; % L2 weight regularization derivative
    end    
    net.layer(l).db = (1/m) * sum(net.layer(l+1).delta, 2); % Bias derivative
    grad = [net.layer(l).dW(:); net.layer(l).db; grad];
  end
  net.layer(1).dW = (1/m) * net.layer(2).delta * net.layer(1).a'; % Weight derivative
  if isfield(net.layer(2), 'reg') && strcmp(net.layer(2).reg, 'L2')
    net.layer(1).dW = net.layer(1).dW + net.lambda * net.layer(1).W; % L2 weight regularization derivative
  end
  net.layer(1).db = (1/m) * sum(net.layer(2).delta, 2); % Bias derivative
  grad = [net.layer(1).dW(:); net.layer(1).db; grad];
end