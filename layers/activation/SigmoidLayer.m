classdef SigmoidLayer < Layer
  % SIGMOIDLAYER SIgmoid activation function layer
  methods (Static)
    % Constructor
    function this = SigmoidLayer(spec)
      this@Layer(spec);
    end
  end
  methods
    % Forward propagation
    function out = forward(this, in)
      out = 1 ./ (1 + exp(-in));
      forward@Layer(this);
    end
    % Backward propagation
    function grad = backward(this, out)
      a = 1 ./ (1 + exp(-out));
      grad = a .* (1 - a);
      backward@Layer(this);
    end
  end
end