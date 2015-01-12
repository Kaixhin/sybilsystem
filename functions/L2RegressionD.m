function costD = L2RegressionD(h, y)
  % l2REGRESSIOND Calculates derivative of half squared error cost
  % h:      Hypothesis
  % y:      Target
  % costD:  Cost derivative
  costD = -(y - h);
end