# Graph

A neural network can be represented as a *flow graph*. Data flows through weighted edges and is transformed into a different feature space by the activation function at a neuron (node).
This is the most general form of a neural network. This library constrains networks such that each node is a layer rather than an individual neuron.
Data then flows, i.e. forward propagates, through the network as a n x m matrix where n is the feature vector length and m is the number of samples.

A feedforward neural network is represented by a *directed acyclic graph*, usually just a chain (graph nodes have a maximum degree of 2) TODO Check term "chain"
The graph abstraction allows the use of BackPropagation Through Structure (BPTS), a generalised version of the backpropagation algorithm for calculating gradients when each node may have more than one parent and/or descendant.
The order in which layers process data for forward and backpropagation can be calculated using a topological sort of the graph.

Recurrent Neural Networks (RNNs) can be *unfolded* for a discrete number of time steps to remove their cycles, and then trained via BPTS; this technique is known as BackPropagation Through Time.
Recursive Neural Networks can therefore be seen as the generalised version of unfolded RNNs.

To clarify this abstraction, a neural network is a graph, a layer is a node and a connection is an edge.

## Usage
nn = new NeuralNetwork()
a = nn.addLayer(84, 'sigmoid')
b = nn.addLayer(10, 'softmax')
a.connectsTo(b)
a.connectedTo
-> [b]
b.connectedFrom
-> [a]
nn.initialize()
nn.order
-> [a b]

## Documentation
### NeuralNetwork
- Constructor()
-- Creates an empty neural network.
- order
-- The topological order of the network. Needs to be calculated after all layers and connections are added.
- addLayer(Int n, String act)
-- Adds a size n layer with activation function act.
- initialize()
-- Adds initial weight matrices to all the layers and calls topologicalSort().
- topologicalSort()
-- Performs a topological sort on the network, storing the order in order.
- setParams(Double[] theta)
-- Sets the network weights (and biases) from theta.
- forwardProp(Double[][] x, *Int/Double[][] y*)
- Performs forward propagation with input x and returns the network output. If provided with target y returns the cost as well.
- backProp(Double[][] x, *Int/Double[][] y*)
- Performs backpropagation with respect to input x and target y.
### Layer
- connectedTo
-- A list of layers that the current layer has a directed edge to.
- connectedFrom
-- A list of layers that the current layer has a directed edge from.
- connectsTo(Layer l)
-- Creates a directed connection from the layer to layer l.
