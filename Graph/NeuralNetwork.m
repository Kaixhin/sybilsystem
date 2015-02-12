classdef NeuralNetwork < handle
  %NEURALNETWORK Neural network class based on the graph data structure
  properties
    Layers = struct
    Order = {}
    UnmarkedNodes = {} % Cell of unmarked nodes
    Loss = '' % TODO Accommodate losses anywhere
    Cost = 0 % Cumulative cost from losses and regularization
    Lambda = 0 % TODO Specify L2 parameter by layer
  end
  methods (Static)
    % Constructor
    function this = NeuralNetwork()
    end
  end
  methods   
    % Adds a layer with a given name and specification
    function addLayer(this, name, spec)
      % Check doesn't exist already
      if (isfield(this.Layers, name))
        error('This layer already exists')
      end
      this.Layers.(name) = Layer(name, spec);
      this.UnmarkedNodes = [name this.UnmarkedNodes]; % Add new layers to unmarked nodes
    end
    
    % Creates a directed connection from layerFrom to layerTo
    function connect(this, nameFrom, nameTo)
      % Check not loop
      if (strcmp(nameFrom, nameTo))
        error('Loops are not allowed')
      end
      % Check not already connected
      if (ismember(nameTo, this.Layers.(nameFrom).ConnectedTo))
        error('Layer %s is already connected to layer %s', nameFrom, nameTo)
      end
      % Otherwise connect
      this.Layers.(nameFrom).ConnectedTo = [this.Layers.(nameFrom).ConnectedTo nameTo];
      this.Layers.(nameTo).ConnectedFrom = [this.Layers.(nameTo).ConnectedFrom nameFrom];
    end
    
    % Initialise once all layers have been added TODO Generalise
    function theta = initialize(this)
      this.topsort(); % Find order for calculations
      theta = [];
      % TODO Don't assume first layer is input and everything is a unary tree
      for l = 2:length(this.Order)
        prevLayer = this.Layers.(this.Order{l-1});
        currLayer = this.Layers.(this.Order{l});
        r = sqrt(6) / sqrt(prevLayer.Size + currLayer.Size); % Choose weights uniformly from the interval [-r, r]
        prevLayer.W = rand([currLayer.Size prevLayer.Size]) * 2 * r - r; % Weights
        prevLayer.b = zeros(currLayer.Size, 1); % Biases
        theta = [theta; prevLayer.W(:); prevLayer.b(:)];
      end 
    end    
    
    % Perform depth-first search topological sort
    function topsort(this)
      while (~isempty(this.UnmarkedNodes))
        n = this.Layers.(this.UnmarkedNodes{1}); % Pick unmarked node n
        visit(n);
      end      
      function visit(n)
        if (strcmp(n.Mark, 'Temporary')) % If marked temp then cyclic
          error('This neural network contains a cycle')
        elseif (strcmp(n.Mark, '')) % If unmarked continue
          n.Mark = 'Temporary';
          this.UnmarkedNodes(find(strcmp(this.UnmarkedNodes, n.Name))) = []; % Remove from unmarked nodes
          for l = 1:length(n.ConnectedTo)
            visit(this.Layers.(n.ConnectedTo{l}));           
          end
          n.Mark = 'Permanent';
          this.Order = [n.Name this.Order];
        end
      end
    end
    
    % Set weights and biases
    function setParams(this, theta)
      prevWbLength = 0;
      for l = 2:length(this.Order)
        prevLayer = this.Layers.(this.Order{l-1});
        currLayer = this.Layers.(this.Order{l});
        WLength = currLayer.Size * prevLayer.Size;
        bLength = currLayer.Size;
        W = theta(1 + prevWbLength:prevWbLength + WLength); % Extract weights
        prevLayer.W = reshape(W, [currLayer.Size prevLayer.Size]); % Set weights    
        prevLayer.b = theta(1 + prevWbLength + WLength:prevWbLength + WLength + bLength); % Extract and set biases
        prevWbLength = prevWbLength + WLength + bLength;
      end
    end
    
    % Calculate network hypothesis
    function h = forwardProp(this, X)
      m = size(X, 2);
      this.Cost = 0; % Reset cost
      this.Layers.(this.Order{1}).a = X; % Input activation
      % TODO Don't assume first layer is input and everything is a unary tree
      for l = 2:length(this.Order)
        prevLayer = this.Layers.(this.Order{l-1});
        currLayer = this.Layers.(this.Order{l});
        currLayer.z = prevLayer.W * prevLayer.a + repmat(prevLayer.b, [1 m]); % Weighted inputs + biases
        fn = str2func(currLayer.Function); % Layer function
        currLayer.a = fn(currLayer.z); % Activation
        if isprop(currLayer, 'Reg') && strcmp(currLayer.Reg, 'L2')
          this.Cost = this.Cost + (this.Lambda/2) * sum(sum(prevLayer.W.^2)); % L2 weight regularization cost
        end
        %{
        if isfield(net.layer(l), 'rho') && isscalar(net.layer(l).rho)
          rho = net.layer(l).rho; % Sparsity parameter
          rhoHat = (1/m) * sum(net.layer(l).a, 2); % Average activations
          net.layer(l).rhoHat = rhoHat;
          KL = rho*log(rho./rhoHat) + (1 - rho)*log((1 - rho)./(1 - rhoHat)); % Kullback-Leibler divergence
          cost = cost + net.beta * sum(KL); % Sparsity cost
        end
        %}
      end
      h = this.Layers.(this.Order{end}).a; % Hypothesis
    end
    
    % Calculate cost and gradient given the target output
    function grad = backProp(this, h, y)
      m = size(h, 2);
      lossFn = str2func(this.Loss); % Loss function
      this.Cost = this.Cost + lossFn(h, y); % Loss cost
      
      grad = [];
      L = length(this.Order);
      lossFnD = str2func(strcat(this.Loss, 'D'));
      this.Layers.(this.Order{L}).delta = lossFnD(h, y); % Loss cost error
      %{
      if isfield(net.layer(L), 'rho') && isscalar(net.layer(L).rho)
        rho = net.layer(L).rho; % Sparsity parameter
        rhoHat = net.layer(L).rhoHat; % Average activations
        net.layer(L).delta = net.layer(L).delta + repmat(net.beta*(-(rho./rhoHat) + ((1 - rho)./(1 - rhoHat))), [1 m]); % Sparsity derivative
      end
      %}
      fnD = str2func(strcat(this.Layers.(this.Order{L}).Function, 'D'));
      this.Layers.(this.Order{L}).delta = this.Layers.(this.Order{L}).delta .* fnD(this.Layers.(this.Order{L}).z); % Error due to weighted inputs
      for l = L-1:-1:2
        nextLayer = this.Layers.(this.Order{l+1});
        currLayer = this.Layers.(this.Order{l});
        currLayer.delta = currLayer.W' * nextLayer.delta; % Error from outputs
        %{
        if isfield(net.layer(l), 'rho') && isscalar(net.layer(l).rho)
          rho = net.layer(l).rho; % Sparsity parameter
          rhoHat = net.layer(l).rhoHat; % Average activations
          net.layer(l).delta = net.layer(l).delta + repmat(net.beta*(-(rho./rhoHat) + ((1 - rho)./(1 - rhoHat))), [1 m]); % Sparsity derivative
        end
        %}
        fnD = str2func(strcat(currLayer.Function, 'D'));
        currLayer.delta = currLayer.delta .* fnD(currLayer.z); % Error due to weighted inputs
        currLayer.dW = (1/m) * nextLayer.delta * currLayer.a'; % Weight derivative
        if isprop(nextLayer, 'Reg') && strcmp(nextLayer.Reg, 'L2')
          currLayer.dW = currLayer.dW + this.Lambda * currLayer.W; % L2 weight regularization derivative
        end
        currLayer.db = (1/m) * sum(nextLayer.delta, 2); % Bias derivative
        grad = [currLayer.dW(:); currLayer.db; grad];
      end
      this.Layers.(this.Order{1}).dW = (1/m) * this.Layers.(this.Order{2}).delta * this.Layers.(this.Order{1}).a'; % Weight derivative
      if isprop(this.Layers.(this.Order{2}), 'Reg') && strcmp(this.Layers.(this.Order{2}).Reg, 'L2')
        this.Layers.(this.Order{1}).dW = this.Layers.(this.Order{1}).dW + this.Lambda * this.Layers.(this.Order{1}).W; % L2 weight regularization derivative
      end
      this.Layers.(this.Order{1}).db = (1/m) * sum(this.Layers.(this.Order{2}).delta, 2); % Bias derivative
      grad = [this.Layers.(this.Order{1}).dW(:); this.Layers.(this.Order{1}).db(:); grad];
    end
  end
end
% TODO Implement Coffman-Graham parallel topsort and events for parallelising backpropagation?