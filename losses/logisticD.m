function lossD = logisticD(h, g)
  % LOGISTICD Calculates derivative of logistic loss
  % h:      Hypothesis
  % g:      Ground truth
  % lossD:  Loss derivative
  lossD = -(g - h);
end