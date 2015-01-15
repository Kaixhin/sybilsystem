function loss = logistic(h, g)
  % LOGISTIC Calculates logistic loss
  % h:    Input
  % g:    Ground truth
  % loss: Loss
  m = size(h, 2);
  loss = -(1/m) * sum(sum(g .* log(h)));
end