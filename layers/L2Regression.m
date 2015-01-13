function [h, cost] = L2Regression(h, y)
  % L2Regression Calculates half squared error cost
  % h:    Hypothesis
  % y:    Target
  % cost: Cost
  m = size(h, 2);
  sqErrCost = sqrt(sum((h - y).^2, 1));
  cost = (1/m) * sum((1/2) * sqErrCost.^2); % Normal cost
end