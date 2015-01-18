function [optTheta, cost] = gradientDescent(fn, theta, options)
  % GRADIENTDESCENT Performs gradient descent
  % fn:       Function that takes theta and returns a cost and gradient for theta
  % theta:    Initial parameters
  % options:  Options
  % optTheta: Optimised parameters
  % cost:     Final cost
  
  alpha = 0.01; % Gradient descent parameter
  v = zeros(size(theta)); % Momentum
  gamma = 0.9; % Momentum parameter
  maxIter = 100; % Maximum amount of iterations
  if (options.maxIter)
    maxIter = options.maxIter;
  end  
  
  [cost, grad] = fn(theta);
  if (options.display)
    disp(cost);
  end
  
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
end