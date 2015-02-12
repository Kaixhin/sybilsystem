function sigmD = sigmoidD(z, a)
  % SIGMOIDD Calculates the derivative of the sigmoid function
  % z:    Input
  % a:    Sigmoid of input
  % sigm: Derivative of sigmoid of input
  sigmD = a .* (1 - a);
end
