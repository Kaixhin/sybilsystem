function rectD = reluD(z, a)
  % RELUD Calculates the derivative of the half rectified function
  % z:      Input
  % a:      Half rectified input
  % rectd:  Derivative of half rectified input
  rectD = a;
  rectD(rectD > 0) = 1;
end