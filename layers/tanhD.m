function tanhDv = tanhD(x)
  % TANHD Calculates the derivative of the tanh function
  % x:      Input
  % tanhDv: Derivative of tanh of input
  tanhDv = 1 - tanh(x).^2;
end