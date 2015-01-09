function cost = halfSquaredErrorCost(h, y)
  % HALFSQUAREDERRORCOST Calculates half squared error cost
  % h:  Hypothesis
  % y:  Target
  m = size(h, 2);
  sqErrCost = sqrt(sum((h - y).^2, 1));
  cost = (1/m) * sum((1/2) * sqErrCost.^2); % Normal cost
end