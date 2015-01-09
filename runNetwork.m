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
  % Set network parameters
  prevWLength = 0;
  for l = 2:L
    WLength = net.layer(l).size * net.layer(l-1).size;
    W = theta(1 + prevWLength:prevWLength + WLength); % Extract weights
    net.layer(l-1).W = reshape(W, [net.layer(l).size net.layer(l-1).size]); % Set weights
    prevWLength = prevWLength + WLength;
  end
  % Forward propagation
  for l = 2:L
    net.layer(l).z = net.layer(l-1).W * net.layer(l-1).a; % Weighted inputs
    fn = str2func(net.layer(l).func); % Layer function
    net.layer(l).a = fn(net.layer(l).z); % Activation
  end
  h = net.layer(L).a; % Hypothesis
  fn = str2func(strcat(net.layer(L).cost, 'Cost')); % Cost function
  cost = fn(h, y); % Cost
  % Backward propagation
  grad = [];
  fnD = str2func(strcat(net.layer(L).func, 'D'));
  net.layer(L).delta = -(y - net.layer(L).a) .* fnD(net.layer(L).z); % Error term TODO Make automatic
  for l = L-1:-1:2
    fnD = str2func(strcat(net.layer(l).func, 'D'));
    net.layer(l).delta = (net.layer(l).W' * net.layer(l+1).delta) .* fnD(net.layer(l).z); % Error term
    net.layer(l).dW = net.layer(l+1).delta * net.layer(l).a';
    grad = [net.layer(l).dW(:); grad];
  end
  net.layer(1).dW = net.layer(2).delta * net.layer(1).a';
  grad = [net.layer(1).dW(:); grad];
end