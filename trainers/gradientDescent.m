function [optTheta, cost] = gradientDescent(fn, theta, options)
  % GRADIENTDESCENT Performs gradient descent
  % fn:       Function that takes theta and returns a cost and gradient for theta
  % theta:    Initial parameters
  % options:  Options
  % optTheta: Optimised parameters
  % cost:     Final cost
  
  alpha = 0.1; % Gradient descent parameter
  %v = zeros(size(theta)); % Momentum
  %gamma = 0; % Momentum parameter
  
  gamma = 0.9; % Decay paramter for moving average
  epsilon = 1e-4; % Constant  
  
  maxIter = 100; % Maximum amount of iterations
  if (options.maxIter)
    maxIter = options.maxIter;
  end
  
  if (options.plot)
    figure
    hold on
  end
    
  %{
  Momentum
  for n = 1:maxIter
    v = gamma*v + alpha*grad;
    theta = theta - v;
    [cost, grad] = fn(theta);    
    if (options.display)
      disp(cost);
    end
  end
  
  v = gamma*v + alpha*grad;
  optTheta = theta - v;
  %}
  
  % AdaDelta
  g = zeros(size(theta));
  s = zeros(size(theta));
  for n = 1:maxIter
    [cost, grad] = fn(theta);
    g = (1 - gamma) * grad.^2 + gamma*g;
    deltaTheta = -alpha*(sqrt(s + epsilon)./sqrt(g + epsilon)) .* grad;
    theta = theta + deltaTheta;
    s = (1 - gamma) * deltaTheta.^2 + gamma*s;
    if (options.display)
      disp(cost);
    end
    if (options.plot)
      plot(n, cost, 'k.')
      pause(0.01)
    end
  end
  
  if (options.plot)
    hold off
  end
  
  optTheta = theta;
end