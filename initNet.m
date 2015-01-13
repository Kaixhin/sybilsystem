function [net theta] = initNetParams(net, inDim, outDim)
  % INITNETPARAMS Initialize network
  % net:    Network
  % inDim:  Input dimension
  % outDim: Output dimension
  % theta:  Network parameters
  
  % Input layer
  net.layer(1).size = inDim;
  
  % Hidden layers
  L = length(net.layer);
  theta = [];
  for l = 2:L-1
    r = sqrt(6) / sqrt(net.layer(l).size + net.layer(l-1).size); % Choose weights uniformly from the interval [-r, r]
    net.layer(l-1).W = rand([net.layer(l).size net.layer(l-1).size]) * 2 * r - r; % Weights
    theta = [theta; net.layer(l-1).W(:)];
  end
  
  % Output layer
  net.layer(L).size = outDim;
  if (strcmp(net.layer(L).func, 'L2Regression'))
    net.layer(L-1).W = eye(net.layer(L-1).size); % Identity matrix TODO Remove this hack
  else
    r = sqrt(6) / sqrt(net.layer(L).size + net.layer(L-1).size); % Choose weights uniformly from the interval [-r, r]
    net.layer(L-1).W = rand([net.layer(L).size net.layer(L-1).size]) * 2 * r - r; % Weights  
  end
  theta = [theta; net.layer(l-1).W(:)];
end