function lossD = logisticD(h, y)
  % LOGISTICD Calculates derivative of logistic loss
  % h:      Hypothesis
  % y:      Ground truth labels (starting at 1)
  % lossD:  Loss derivative
  m = size(h, 2);
  g = full(sparse(y, 1:m, 1)); % Convert to Kronecker delta representation
  lossD = -(g - h);
end