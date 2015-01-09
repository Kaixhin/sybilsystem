function sigm = sigmoid(x)
  % SIGMOID Calculates the sigmoid function
  % x:    Input
  % sigm: Sigmoid of input
  sigm = 1 ./ (1 + exp(-x));
end