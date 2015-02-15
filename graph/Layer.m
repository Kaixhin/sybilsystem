% A layer takes a set of inputs and returns a set of outputs on forward prop


classdef Layer < dynamicprops
  %LAYER Neural network layer based on the vertex of a graph
  properties
    Name % Unique name used for identification and connecting layers
    Function % Activation function
    Size % Input size
    W % Weights
    b % Biases
    z % Weighted inputs
    a % Activation
    delta % Error
    dW % Weight derivatives
    db % Bias derivatives
    ConnectedTo = {} % Cell of outgoing edges
    ConnectedFrom = {} % Cell of incoming edges
    Mark = '' % Mark used in graphing algorithms
  end
  methods (Static)
    % Constructor
    function this = Layer(name, spec)
      this.Name = name;
      this.Function = spec.Function;
      this.Size = spec.Size;
      % Add optional properties
      % Add weight regularization
      if (isfield(spec, 'Reg'))
        this.addprop('Reg');
        this.Reg = spec.Reg;
      end
      % Add sparsity parameter
      if (isfield(spec, 'Rho'))
        this.addprop('Rho');
        this.Rho = spec.Rho;
      end
    end
  end
end