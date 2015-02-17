% A layer takes a set of inputs and returns a set of outputs on forward prop

classdef Layer < dynamicprops
  %LAYER Neural network layer based on the vertex of a graph
  properties
    Name % Unique name used for identification and connecting layers
    In = {} % Cell of incoming blobs (may be empty)
    Out = {} % Cell of outgoing blobs (may be empty)
    Param = {} % Cell of internal parameters (may be empty)
    %Size % Should work out from inputs
    %W % Weights
    %b % Biases
    %z % Weighted inputs
    %a % Activation
    %delta % Error
    %dW % Weight derivatives
    %db % Bias derivatives
    Mark = '' % Mark used in graphing algorithms
  end
  events
    Forward % Triggered on forward propagation
    Backward % Triggered on backward propagation
  end
  methods (Static)
    % Constructor
    function this = Layer(spec)
      this.Name = spec.name;
      if (isfield(spec, 'in'))
        this.In = spec.in; % Set inputs
      end
      if (isfield(spec, 'out'))
        this.Out = spec.out; % Set outputs
      end      
      %{
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
      %}
    end
  end
  methods
    % Forward propagation
    function forward(this)
      notify(this, 'Forward'); % Notify next nodes
    end
    % Backward propagation
    function backward(this)
      notify(this, 'Backward'); % Notify previous nodes
    end
  end
end