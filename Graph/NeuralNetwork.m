classdef NeuralNetwork < handle
  %NEURALNETWORK Neural network class based on the graph data structure
  properties (SetAccess = private)
    Layers = struct
    Order = {}
    UnmarkedNodes = {} % Cell of unmarked nodes
  end
  methods (Static)
    % Constructor
    function this = NeuralNetwork()
    end
  end
  methods   
    % Adds a layer with a given name and specification
    function addLayer(this, name, spec)
      this.Layers.(name) = Layer(name, spec);
      this.UnmarkedNodes = [name this.UnmarkedNodes]; % Add new layers to unmarked nodes
    end
    % Creates a directed connection from layerFrom to layerTo
    function connect(this, nameFrom, nameTo)
      % Check not loop
      if (nameFrom == nameTo)
        error('Loops are not allowed')
      end
      % Check not already connected
      if (ismember(nameTo, this.Layers.(nameFrom).ConnectedTo))
        error('%s is already connected to %s', nameFrom, nameTo)
      end
      % Otherwise connect
      this.Layers.(nameFrom).ConnectedTo = [this.Layers.(nameFrom).ConnectedTo nameTo];
      this.Layers.(nameTo).ConnectedFrom = [this.Layers.(nameTo).ConnectedFrom nameFrom];
    end
    % Initialise once all layers have been added
    function initialize(this)
      this.topsort(); % Find order for calculations
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
          for k = 1:length(n.ConnectedTo)
            visit(this.Layers.(n.ConnectedTo{k}));           
          end
          n.Mark = 'Permanent';
          this.Order = [n.Name this.Order];
        end
      end
    end
    function setParams(this, theta)
    end
    function forwardProp(this, x)
    end
    function backProp(this, x, y)
    end
  end  
end
% TODO Implement Coffman-Graham parallel topsort and events for parallelising backpropagation?