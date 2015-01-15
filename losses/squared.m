function loss = squared(h, y)
  % SQUARED Calculates squared loss
  % h:    Hypothesis
  % y:    Target
  % loss: Loss
  m = size(h, 2);
  sqErrCost = sqrt(sum((h - y).^2, 1));
  loss = (1/m) * sum((1/2) * sqErrCost.^2);
end