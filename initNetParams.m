function [net theta] = initNetParams(net)
  % INITNETPARAMS Initialize network parameters
  % net:    Network
  % theta:  Network parameters
  theta = [];
  for l = 2:length(net.layer)
    if (strcmp(net.layer(l).func, 'L2Regression'))
      net.layer(l-1).W = eye(net.layer(l-1).size); % Identity matrix
    else
      r = sqrt(6) / sqrt(net.layer(l).size + net.layer(l-1).size); % Choose weights uniformly from the interval [-r, r]
      net.layer(l-1).W = rand([net.layer(l).size net.layer(l-1).size]) * 2 * r - r; % Weights
    end
      theta = [theta; net.layer(l-1).W(:)];
  end
end