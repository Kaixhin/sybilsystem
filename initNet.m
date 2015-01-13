function [net theta] = initNet(net, inDim, outDim)
  % INITNET Initialize network
  % net:    Network
  % inDim:  Input dimension
  % outDim: Output dimension
  % theta:  Network parameters
  
  L = length(net.layer);
  net.layer(1).size = inDim; % Input layer
  net.layer(L).size = outDim; % Output layer

  theta = [];
  for l = 2:L
    r = sqrt(6) / sqrt(net.layer(l).size + net.layer(l-1).size); % Choose weights uniformly from the interval [-r, r]
    net.layer(l-1).W = rand([net.layer(l).size net.layer(l-1).size]) * 2 * r - r; % Weights
    net.layer(l-1).b = zeros(net.layer(l).size, 1); % Biases
    theta = [theta; net.layer(l-1).W(:); net.layer(l-1).b];
  end
end