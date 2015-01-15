function sigmD = sigmoidD(z)
  % SIGMOIDD Calculates the derivative of the sigmoid function
  % z:    Input
  % sigm: Derivative of sigmoid of input
  sigm = 1 ./ (1 + exp(-z));
  sigmD = sigm .* (1 - sigm);
end