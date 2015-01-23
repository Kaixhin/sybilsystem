function [optTheta, cost] = gradientDescent(fn, theta, options)
  % GRADIENTDESCENT Performs gradient descent
  % fn:       Function that takes theta and returns a cost and gradient for theta
  % theta:    Initial parameters
  % options:  Options
  % optTheta: Optimised parameters
  % cost:     Final cost
  tic

  % Parameters
  alpha = 0.8; % Learning rate
  gamma = 0.9; % Momentum parameter
  gradHist = zeros(size(theta)); % Adaptive historical gradient
  epsilon = 1e-6; % Constant for numerical stability
  rho = 0.95; % Decay rate for moving average
  deltaHist = zeros(size(theta)); % AdaDelta history
  thetaHist = theta; % Past state for convergence check
  % TODO Add gradient clipping for gradient cliffs (esp. for RNNs)
  % Options
  method = 'adadelta'; % Default method (sgd/nesterov/adagrad/adadelta)
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
    if strcmp(method, 'sgd') % Gradient descent with momentum
      adjGrad = gamma*gradHist - alpha*grad;
      gradHist = adjGrad;
      theta = theta + adjGrad;
    elseif strcmp(method, 'nesterov') % Nesterov accelerated gradient TODO Check implementation
      adjGrad = gradHist;
      gradHist = gamma*gradHist + alpha*grad;
      adjGrad = gamma*adjGrad - (1 - gamma)*gradHist;
      theta = theta + adjGrad;
    elseif strcmp(method, 'adagrad') % Adapt gradient for rare features
      gradHist = gradHist + grad.^2;
      adjGrad = -alpha * grad./(sqrt(gradHist) + epsilon);
      theta = theta + adjGrad;
    elseif strcmp(method, 'adadelta') % Adapt gradient per dimension
      gradHist = (1 - rho)*grad.^2 + rho*gradHist;
      adjGrad = -alpha * (sqrt(deltaHist + epsilon)./sqrt(gradHist + epsilon)) .* grad;
      theta = theta + adjGrad;
      deltaHist = (1 - rho)*adjGrad.^2 + rho*deltaHist;
    end
       
    if (dispCost)
      fprintf('Iteration: %d\t\t\tCost: %f\n', n, cost);
    end
    if (dispPlot)
      plot(n, cost, 'k.')
      pause(0.01)
    end
    
    if (mod(n, 10) == 0) % Check for convergence every 10 iterations
      if (norm(theta - thetaHist) < epsilon)
        break
      else
        thetaHist = theta;
      end
    end   
  end  

  optTheta = theta;
  toc
end
