function costD = softmaxD(h, g)
  % SOFTMAXD Calculates derivative of softmax cost
  % h:      Hypothesis
  % g:      Ground truth
  % costD:  Cost derivative
  costD = -(g - h);
end