function softD = softplusD(z, ~)
  % SOFTPLUSD Calculates the derivative of the softplus function
  % z:      Input
  % a:      Softplus of input
  % softD:  Derivative of softmax of input
  softD = sigmoid(z);
end