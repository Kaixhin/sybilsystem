function lossD = squaredD(h, y)
  % SQUAREDD Calculates derivative of squared loss
  % h:      Hypothesis
  % y:      Target
  % lossD:  Loss derivative
  lossD = -(y - h);
end