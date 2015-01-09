function sigmD = sigmoid(x)
  % SIGMOID Calculates the derivative of the sigmoid function
  % x:    Input
  % sigm: Derivative of sigmoid of input
  sigm = 1 ./ (1 + exp(-x));
  sigmD = sigm .* (1 - sigm);
end