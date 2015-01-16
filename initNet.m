function [net theta] = initNet(net, X, y)
  % INITNET Initialize network
  % net:    Network
  % X:      Input 
  % y:      Target
  % theta:  Network parameters
  
  L = length(net.layer);
  net.layer(1).size = size(X, 1); % Input layer
  if strcmp(net.loss, 'logistic') || strcmp(net.loss, 'hinge')
    net.layer(L).size = max(y); % Output layer with target as labels
  else
    net.layer(L).size = size(y, 1); % Output layer matching target
  end
    

  theta = [];
  for l = 2:L
    r = sqrt(6) / sqrt(net.layer(l).size + net.layer(l-1).size); % Choose weights uniformly from the interval [-r, r]
    net.layer(l-1).W = rand([net.layer(l).size net.layer(l-1).size]) * 2 * r - r; % Weights
    net.layer(l-1).b = zeros(net.layer(l).size, 1); % Biases
    theta = [theta; net.layer(l-1).W(:); net.layer(l-1).b];
  end
end