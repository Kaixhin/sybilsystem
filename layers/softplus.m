function soft = softplus(z)
  % SOFTPLUS Calculates the softplus function
  % z:    Input
  % soft: Softplus of input
  soft = log(1 + exp(z));
end