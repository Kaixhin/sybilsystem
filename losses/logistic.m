function loss = logistic(h, y)
  % LOGISTIC Calculates logistic loss
  % h:    Input
  % y:    Ground truth labels (starting at 1)
  % loss: Loss
  m = size(h, 2);
  g = full(sparse(y, 1:m, 1)); % Convert to Kronecker delta representation
  loss = -(1/m) * sum(sum(g .* log(h)));
end