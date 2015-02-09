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
nn = NeuralNetwork();
nn.addLayer('a', struct('Function', 'sigmoid', 'Size', 84, 'Reg', 'L2', 'Rho', 0.1))
nn.addLayer('b', struct('Function', 'softmax', 'Size', 10, 'Reg', 'L2'))
nn.connect('a', 'b')
nn.connect('a', 'a')
nn.connect('b', 'a')
nn.initialize()
nn