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
  net.layer(1).a = X; % Input activation
  for l = 2:L-1
    net.layer(l).z = net.layer(l-1).W * net.layer(l-1).a; % Weighted inputs
    fn = str2func(net.layer(l).func); % Layer function
    net.layer(l).a = fn(net.layer(l).z); % Activation
  end
  % TODO Remove assumption that last layer is losses/costs
  net.layer(L).z = net.layer(L-1).W * net.layer(L-1).a; % Weighted inputs
  fn = str2func(net.layer(L).func); % Layer function
  [net.layer(L).a, cost] = fn(net.layer(L).z, y); % Activation
  h = net.layer(L).a;
  
  % Backward propagation
  grad = [];
  % TODO Remove assumption that last layer is losses/costs
  fnD = str2func(strcat(net.layer(L).func, 'D'));
  net.layer(L).delta = fnD(h, y); % Cost derivative
  for l = L-1:-1:2
    fnD = str2func(strcat(net.layer(l).func, 'D'));
    net.layer(l).delta = (net.layer(l).W' * net.layer(l+1).delta) .* fnD(net.layer(l).z); % Error term TODO Deal with softmax
    net.layer(l).dW = net.layer(l+1).delta * net.layer(l).a'; % TODO Fix this for L2Regression
    grad = [net.layer(l).dW(:); grad];
  end
  net.layer(1).dW = net.layer(2).delta * net.layer(1).a';
  grad = [net.layer(1).dW(:); grad];
end