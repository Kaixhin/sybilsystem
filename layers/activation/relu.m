function rect = relu(z)
  % RELU Calculates the half rectified function
  % z:    Input
  % rect: Half rectified input
  rect = max(0, z);
end