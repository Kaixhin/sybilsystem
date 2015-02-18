classdef DataLayer < Layer
  % DATALAYER Data input layer
  properties
    Source = {} % Source of data
  end
  methods (Static)
    % Constructor
    function this = DataLayer(spec)
      this@Layer(spec);
      this.Out.data
      if (strcmp(spec.source, 'MNIST'))
        this.Out{1}.a = loadMNISTImages(['data' filesep 'train-images.idx3-ubyte']);
      end
    end
  end
  methods
    % Forward propagation
    function [data, label] = forward(this)
      data = 
      label = loadMNISTLabels(['data' filesep 'train-labels.idx1-ubyte']);
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