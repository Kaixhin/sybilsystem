function [condProb, cost] = softmax(h, g)
  % SOFTMAX Calculates the softmax (normalized exponential) function
  % h:        Input
  % g:        Ground truth
  % condProb: Conditional probabilities
  % cost:     Cost
  condProb = exp(bsxfun(@minus, h, max(h, [], 1))); % Prevent overflow
  condProb = bsxfun(@rdivide, condProb, sum(condProb)); % Normalise for probabilities
  m = size(h, 2);
  cost = -(1/m) * sum(sum(g .* log(h)));
end