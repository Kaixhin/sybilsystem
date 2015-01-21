function condProb = softmax(z)
  % SOFTMAX Calculates the softmax (normalized exponential) function
  % z:        Input
  % condProb: Conditional probabilities
  condProb = exp(bsxfun(@minus, z, max(z, [], 1))); % Prevent over/underflow
  condProb = bsxfun(@rdivide, condProb, sum(condProb)); % Normalise for probabilities
end
