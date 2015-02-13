function tanhDv = tanhD(~, a)
  % TANHD Calculates the derivative of the tanh function
  % z:      Input
  % a:      Tanh of input
  % tanhDv: Derivative of tanh of input
  tanhDv = 1 - a.^2;
end
