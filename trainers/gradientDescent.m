function [optTheta, cost] = gradientDescent(fn, theta, options)
  % GRADIENTDESCENT Performs gradient descent
  % fn:       Function that takes theta and returns a cost and gradient for theta
  % theta:    Initial parameters
  % options:  Options
  % optTheta: Optimised parameters
  % cost:     Final cost

  % Parameters
  alpha = 0.01; % Learning rate
  %v = zeros(size(theta)); % Momentum
  %gamma = 0; % Momentum parameter
  epsilon = 1e-6; % Constant for numerical stability
  gradHist = zeros(size(theta)); % Adaptive historical gradient  
  gamma = 0.9; % Decay paramter for moving average
  s = zeros(size(theta)); % AdaDelta state
  % Options
  method = 'adagrad'; % Default method
  if (isfield(options, 'methods'))
    method = options.method;
  end
  maxIter = 100; % Maximum amount of iterations
  if (isfield(options, 'maxIter'))
    maxIter = options.maxIter;
  end
  dispCost = false;
  if (isfield(options, 'display'))
    dispCost = options.display;
  end
  dispPlot = false;
  if (isfield(options, 'plot'))
    dispPlot = options.plot;
  end
  if (dispPlot)
    figure
    hold on
  end
  
  for n = 1:maxIter
    [cost, grad] = fn(theta);
    if strcmp(method, 'adagrad') % Adapt gradient for rare features
      gradHist = gradHist + grad.^2;
      adjGrad = grad./(sqrt(gradHist) + epsilon);
      theta = theta - alpha*adjGrad;
    elseif strcmp(method, 'adadelta')
      gradHist = (1 - gamma) * grad.^2 + gamma*gradHist;
      adjGrad = -alpha*(sqrt(s + epsilon)./sqrt(gradHist + epsilon)) .* grad;
      theta = theta + adjGrad;
      s = (1 - gamma) * adjGrad.^2 + gamma*s;
    end
       
    if (dispCost)
      disp(cost);
    end
    if (dispPlot)
      plot(n, cost, 'k.')
      pause(0.01)
    end
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
  %{
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
  %}
  
  if (dispPlot)
    hold off
  end
  
  optTheta = theta;
end