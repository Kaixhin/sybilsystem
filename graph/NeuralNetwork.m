classdef NeuralNetwork < handle
  %NEURALNETWORK Neural network class based on the graph data structure
  properties
    Layers = {} % Cell of layers
  end
  methods (Static)
    % Constructor
    function this = NeuralNetwork()
    end
  end
  methods   
    % Adds a layer with a given name and specification
    function addLayer(this, spec)
      % Check doesn't exist already
      if (isfield(this.Layers, spec.name))
        error('This layer already exists')
      end
      layerConstructor = str2func(strcat(spec.type, 'Layer'));
      this.Layers.(spec.name) = layerConstructor(spec);
    end
  end
end