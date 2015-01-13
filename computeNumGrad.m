function numgrad = computeNumGrad(J, theta)
  % COMPUTENUMGRAD Computes numerically the gradient of function J with parameters theta
  EPSILON = 10e-4;
  n = size(theta);
  numgrad = zeros([n 1]);
  for i = 1:n
    eVector = zeros(n);
    eVector(i) = eVector(i) + EPSILON;
    numgrad(i) = (J(theta + eVector) - J(theta - eVector)) / (2 * EPSILON);
  end
end