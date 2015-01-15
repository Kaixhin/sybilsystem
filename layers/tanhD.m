function tanhDv = tanhD(z)
  % TANHD Calculates the derivative of the tanh function
  % z:      Input
  % tanhDv: Derivative of tanh of input
  tanhDv = 1 - tanh(z).^2;
end