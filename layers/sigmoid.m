function sigm = sigmoid(z)
  % SIGMOID Calculates the sigmoid function
  % z:    Input
  % sigm: Sigmoid of input
  sigm = 1 ./ (1 + exp(-z));
end